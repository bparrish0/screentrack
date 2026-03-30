use anyhow::{Context, Result};
use base64::Engine as _;
use image::DynamicImage;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::Cursor;
use std::process::Command;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrResult {
    pub text: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OcrBackend {
    AppleVision,
    LlmVision,
    BenchmarkBothPreferLlm,
}

impl Default for OcrBackend {
    fn default() -> Self {
        Self::AppleVision
    }
}

fn default_llm_ocr_timeout_secs() -> u64 {
    90
}

fn default_llm_ocr_max_tokens() -> u32 {
    4096
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmOcrConfig {
    pub base_url: String,
    pub model: String,
    pub api_key: Option<String>,
    #[serde(default = "default_llm_ocr_timeout_secs")]
    pub timeout_secs: u64,
    #[serde(default = "default_llm_ocr_max_tokens")]
    pub max_tokens: u32,
    /// Optional custom OCR prompt.
    pub prompt: Option<String>,
}

const DEFAULT_LLM_OCR_PROMPT: &str = "You are an OCR engine. Extract all visible text exactly as shown. Preserve line breaks. Do not summarize or explain.";

/// Run Apple Vision OCR on an image by saving it to a temp file and using a Swift helper.
/// This is a simpler approach than FFI to the Vision framework directly.
///
/// For a production build, we'd use cidre or objc2 bindings to VNRecognizeTextRequest.
/// This subprocess approach works as an initial implementation.
pub fn ocr_image(image: &DynamicImage) -> Result<OcrResult> {
    // Save image to a temp file
    let tmp = std::env::temp_dir().join("screentrack_ocr_tmp.png");
    image.save(&tmp)?;

    // Use the macOS `shortcuts` CLI or a small Swift script for Vision OCR
    // For now, we use a bundled Swift script
    let script_path = std::env::current_exe()?
        .parent()
        .unwrap()
        .join("ocr_helper.swift");

    // If the helper script doesn't exist, try the built-in screencapture + Vision approach
    if !script_path.exists() {
        return ocr_via_shortcut(&tmp);
    }

    let output = Command::new("swift").arg(&script_path).arg(&tmp).output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("OCR helper failed: {stderr}");
    }

    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    std::fs::remove_file(&tmp).ok();

    Ok(OcrResult {
        text,
        confidence: 0.9, // Vision framework doesn't easily expose aggregate confidence
    })
}

fn image_to_data_url(image: &DynamicImage) -> Result<String> {
    let mut cursor = Cursor::new(Vec::new());
    image
        .write_to(&mut cursor, image::ImageFormat::Png)
        .context("Failed to encode image as PNG")?;
    let encoded = base64::engine::general_purpose::STANDARD.encode(cursor.into_inner());
    Ok(format!("data:image/png;base64,{encoded}"))
}

fn extract_message_content_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.trim().to_string(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Analyze an image with a vision-capable OpenAI-compatible model.
/// For OCR, call this with `prompt=None` or a custom OCR prompt.
pub fn analyze_image_with_llm(
    image: &DynamicImage,
    config: &LlmOcrConfig,
    prompt: Option<&str>,
) -> Result<OcrResult> {
    let data_url = image_to_data_url(image)?;
    let prompt = prompt
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .or_else(|| config.prompt.as_deref())
        .unwrap_or(DEFAULT_LLM_OCR_PROMPT);

    let base = config.base_url.trim_end_matches('/');
    let url = format!("{base}/v1/chat/completions");
    let payload = serde_json::json!({
        "model": config.model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]
        }],
        "temperature": 0.0,
        "max_tokens": config.max_tokens
    });

    let client = Client::builder()
        .timeout(Duration::from_secs(config.timeout_secs))
        .build()
        .context("Failed to build LLM OCR HTTP client")?;
    let mut req = client.post(&url).json(&payload);
    if let Some(ref key) = config.api_key {
        req = req.bearer_auth(key);
    }

    let response = req
        .send()
        .context("Failed to connect to LLM OCR endpoint")?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        anyhow::bail!("LLM OCR API error {status}: {body}");
    }

    let value: Value = response
        .json()
        .context("Failed to parse LLM OCR response JSON")?;
    let content = value.pointer("/choices/0/message/content");
    let text = content
        .map(extract_message_content_text)
        .unwrap_or_default();
    let text = text.trim().to_string();
    if text.is_empty() {
        anyhow::bail!("LLM OCR returned empty content");
    }

    Ok(OcrResult {
        text,
        confidence: 0.7, // Heuristic confidence for text extracted via LLM vision.
    })
}

/// OCR via a vision-capable OpenAI-compatible model endpoint.
pub fn ocr_image_llm(image: &DynamicImage, config: &LlmOcrConfig) -> Result<OcrResult> {
    analyze_image_with_llm(image, config, None)
}

/// Fallback: use macOS built-in text recognition via a small inline Swift subprocess.
fn ocr_via_shortcut(image_path: &std::path::Path) -> Result<OcrResult> {
    let swift_code = format!(
        r#"
import Vision
import AppKit

let url = URL(fileURLWithPath: "{}")
guard let image = NSImage(contentsOf: url),
      let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {{
    exit(1)
}}

let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = true

let handler = VNImageRequestHandler(cgImage: cgImage)
try handler.perform([request])

guard let observations = request.results else {{ exit(0) }}
for observation in observations {{
    if let candidate = observation.topCandidates(1).first {{
        print(candidate.string)
    }}
}}
"#,
        image_path.display()
    );

    let output = Command::new("swift").arg("-e").arg(&swift_code).output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Vision OCR failed: {stderr}");
    }

    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();

    Ok(OcrResult {
        text,
        confidence: 0.9,
    })
}
