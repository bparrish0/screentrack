use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Configuration for the LLM API client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub base_url: String,
    pub model: String,
    pub api_key: Option<String>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub timeout_secs: u64,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:8080".into(),
            model: String::new(),
            api_key: None,
            max_tokens: 16384,
            temperature: 0.3,
            timeout_secs: 300,
        }
    }
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    max_tokens: u32,
    temperature: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
    usage: Option<ChatUsage>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Debug, Deserialize)]
struct ChatMessageResponse {
    content: String,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatUsage {
    prompt_tokens: Option<usize>,
    completion_tokens: Option<usize>,
}

/// Token usage stats from an LLM response.
#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

pub struct LlmClient {
    config: LlmConfig,
    http: reqwest::Client,
}

impl LlmClient {
    pub fn new(config: LlmConfig) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to build HTTP client");

        Self { config, http }
    }

    /// Get the base URL of the LLM endpoint.
    pub fn base_url(&self) -> Option<&str> {
        Some(&self.config.base_url)
    }

    /// Send a chat completion request and return the response text.
    pub async fn chat(&self, system: &str, user: &str) -> Result<String> {
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: system.into(),
            },
            ChatMessage {
                role: "user".into(),
                content: user.into(),
            },
        ];

        self.chat_messages(&messages).await
    }

    /// Send a chat completion with full message history.
    pub async fn chat_messages(&self, messages: &[ChatMessage]) -> Result<String> {
        let (text, _usage) = self.chat_messages_with_usage(messages).await?;
        Ok(text)
    }

    /// Send a chat completion and return both the response text and token usage.
    pub async fn chat_messages_with_usage(
        &self,
        messages: &[ChatMessage],
    ) -> Result<(String, TokenUsage)> {
        let url = format!("{}/v1/chat/completions", self.config.base_url);

        let request = ChatRequest {
            model: self.config.model.clone(),
            messages: messages.to_vec(),
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
        };

        let mut req = self.http.post(&url).json(&request);
        if let Some(ref key) = self.config.api_key {
            req = req.bearer_auth(key);
        }

        let response = req
            .send()
            .await
            .context("Failed to connect to LLM server")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("LLM API error {status}: {body}");
        }

        let chat_response: ChatResponse = response
            .json()
            .await
            .context("Failed to parse LLM response")?;

        let usage = chat_response
            .usage
            .map(|u| TokenUsage {
                prompt_tokens: u.prompt_tokens.unwrap_or(0),
                completion_tokens: u.completion_tokens.unwrap_or(0),
            })
            .unwrap_or_default();

        let text = chat_response
            .choices
            .into_iter()
            .next()
            .map(|c| {
                // Strip residual thinking tags from reasoning models
                let text = c.message.content;
                let text = text.strip_prefix("</think>").unwrap_or(&text);
                text.trim_start().to_string()
            })
            .context("LLM returned no choices")?;

        Ok((text, usage))
    }

    /// Send a simple chat and return both response text and token usage.
    pub async fn chat_with_usage(
        &self,
        system: &str,
        user: &str,
    ) -> Result<(String, TokenUsage)> {
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: system.into(),
            },
            ChatMessage {
                role: "user".into(),
                content: user.into(),
            },
        ];
        self.chat_messages_with_usage(&messages).await
    }
}
