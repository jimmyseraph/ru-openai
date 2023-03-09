
use std::fmt::{Display, Formatter};
use std::fs;
use crate::configuration::Configuration;
use reqwest::StatusCode;
use serde_derive::{Deserialize, Serialize};
use reqwest::{Method, multipart::Part};
use tracing::*;

#[derive(Deserialize, Serialize, Debug)]
pub struct ErrorInfo {
    pub message: String,
    #[serde(rename = "type")]
    pub message_type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct ReturnErrorType {
    pub error: ErrorInfo,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct OpenAIApiError {
    pub code: i32,
    pub error: ErrorInfo,
}

impl OpenAIApiError {
    pub fn new(code: i32, error: ErrorInfo) -> Self {
        Self { code, error }
    }

    pub fn from(error: reqwest::Error) -> Self {
        let code = error.status().unwrap_or(StatusCode::INTERNAL_SERVER_ERROR).as_u16() as i32;
        let error = ErrorInfo {
            message: error.to_string(),
            message_type: "request error".to_string(),
            param: None,
            code: None,
        };
        Self::new(code, error)
    }
}

pub type Error = reqwest::Error;


#[derive(Deserialize, Serialize, Debug)]
pub struct Permission {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub allow_create_engine: bool,
    pub allow_sampling: bool,
    pub allow_logprobs: bool,
    pub allow_search_indices: bool,
    pub allow_view: bool,
    pub allow_fine_tuning: bool,
    pub organization: String,
    pub group: Option<String>,
    pub is_blocking: bool,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub owned_by: String,
    pub permission: Vec<Permission>,
    pub root: String,
    pub parent: Option<String>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct ListModelsResponse {
    pub data: Vec<ModelInfo>,
    pub object: String,
}

pub type RetrieveModelResponse = ModelInfo;

#[derive(Deserialize, Serialize, Debug, Default)]
pub struct CreateCompletionRequest {
    /// ID of the model to use. 
    /// You can use the List models API to see all of your available models, or see our Model overview for descriptions of them.
    pub model: String,
    /// The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays.
    /// Note that <|endoftext|> is the document separator that the model sees during training, so if a prompt is not specified the model will generate as if from the beginning of a new document.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<Vec<String>>,
    /// The suffix that comes after a completion of inserted text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
    /// The maximum number of tokens to generate in the completion.
    /// The token count of your prompt plus max_tokens cannot exceed the model's context length. 
    /// Most models have a context length of 2048 tokens (except for the newest models, which support 4096).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    /// We generally recommend altering this or top_p but not both.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. 
    /// So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    /// We generally recommend altering this or temperature but not both.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// How many completions to generate for each prompt.
    /// Note: Because this parameter generates many completions, it can quickly consume your token quota. 
    /// Use carefully and ensure that you have reasonable settings for max_tokens and stop.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u16>,
    /// Whether to stream back partial progress. If set, tokens will be sent as data-only server-sent events as they become available, with the stream terminated by a data: [DONE] message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Include the log probabilities on the logprobs most likely tokens, as well the chosen tokens. For example, if logprobs is 5, the API will return a list of the 5 most likely tokens. 
    /// The API will always return the logprob of the sampled token, so there may be up to logprobs+1 elements in the response.
    /// The maximum value for logprobs is 5. If you need more than this, please contact us through our Help center and describe your use case.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<i16>,
    /// Echo back the prompt in addition to the completion
    #[serde(skip_serializing_if = "Option::is_none")]
    pub echo: Option<bool>,
    /// Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Generates best_of completions server-side and returns the "best" (the one with the highest log probability per token). Results cannot be streamed.
    /// When used with n, best_of controls the number of candidate completions and n specifies how many to return – best_of must be greater than n.
    /// Note: Because this parameter generates many completions, it can quickly consume your token quota. Use carefully and ensure that you have reasonable settings for max_tokens and stop.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_of: Option<u16>,
    /// Modify the likelihood of specified tokens appearing in the completion.
    /// Accepts a json object that maps tokens (specified by their token ID in the GPT tokenizer) to an associated bias value from -100 to 100. 
    /// You can use this tokenizer tool (which works for both GPT-2 and GPT-3) to convert text to token IDs. 
    /// Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.
    /// As an example, you can pass {"50256": -100} to prevent the <|endoftext|> token from being generated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<serde_json::Value>,
    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. 
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct CreateCompletionResponseChoice {
    pub text: String,
    pub index: i64,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: String,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct Usage {
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub total_tokens: i64,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct CreateCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CreateCompletionResponseChoice>,
    pub usage: Usage,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct ChatFormat {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize, Serialize, Debug, Default)]
pub struct CreateChatCompletionRequest {
    /// ID of the model to use. Currently, only gpt-3.5-turbo and gpt-3.5-turbo-0301 are supported.
    pub model: String,
    /// The messages to generate chat completions for, in the chat format.
    pub messages: Vec<ChatFormat>,
    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    /// We generally recommend altering this or top_p but not both.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    /// We generally recommend altering this or temperature but not both.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// How many chat completion choices to generate for each input message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u16>,
    /// If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as data-only server-sent events as they become available, with the stream terminated by a data: [DONE] message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Up to 4 sequences where the API will stop generating further tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// The maximum number of tokens allowed for the generated answer. By default, the number of tokens the model can return will be (4096 - prompt tokens).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Modify the likelihood of specified tokens appearing in the completion.
    /// Accepts a json object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100. 
    /// Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<serde_json::Value>,
    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct CreateChatCompletionResponseChoice {
    pub message: ChatFormat,
    pub index: i64,
    pub finish_reason: String,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct CreateChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub choices: Vec<CreateChatCompletionResponseChoice>,
    pub usage: Usage,
}

#[derive(Deserialize, Serialize, Debug, Default)]
pub struct CreateEditRequest {
    /// ID of the model to use. You can use the text-davinci-edit-001 or code-davinci-edit-001 model with this endpoint
    pub model: String,
    /// The input text to use as a starting point for the edit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
    /// The instruction that tells the model how to edit the prompt.
    pub instruction: String,
    /// How many edits to generate for the input and instruction.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u16>,
    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    /// We generally recommend altering this or top_p but not both.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. 
    /// So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    /// We generally recommend altering this or temperature but not both.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct CreateEditResponseChoice {
    pub index: i64,
    pub text: String,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct CreateEditResponse {
    pub object: String,
    pub created: i64,
    pub choices: Vec<CreateEditResponseChoice>,
    pub usage: Usage,
}

#[derive(Deserialize, Serialize, Debug)]
pub enum ImageFormat {
    #[serde(rename = "url")]
    URL,
    #[serde(rename = "b64_json")]
    B64JSON,
}

impl Display for ImageFormat {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ImageFormat::URL => write!(f, "url"),
            ImageFormat::B64JSON => write!(f, "b64_json"),
        }
    }
    
}

#[derive(Deserialize, Serialize, Debug, Default)]
pub struct CreateImageRequest {
    /// A text description of the desired image(s). The maximum length is 1000 characters.
    pub prompt: String,
    /// The number of images to generate. Must be between 1 and 10.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u16>,
    /// The size of the generated images. Must be one of 256x256, 512x512, or 1024x1024.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    /// The format in which the generated images are returned. Must be one of ImageFormat::URL or ImageFormat::B64JSON.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ImageFormat>,
    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub enum CreateImageResponseData {
    #[serde(rename = "url")]
    Url(String),
    #[serde(rename = "b64_json")]
    B64Json(String),
}

#[derive(Deserialize, Serialize, Debug)]
pub struct CreateImageResponse {
    pub created: i64,
    pub data: Vec<CreateImageResponseData>,
}

#[derive(Deserialize, Serialize, Debug, Default)]
pub struct CreateImageEditRequest {
    /// The image to edit. Must be a valid PNG file, less than 4MB, and square. 
    /// If mask is not provided, image must have transparency, which will be used as the mask.
    pub image: String,
    /// An additional image whose fully transparent areas (e.g. where alpha is zero) indicate where image should be edited. Must be a valid PNG file, less than 4MB, and have the same dimensions as image.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask: Option<String>,
    /// A text description of the desired image(s). The maximum length is 1000 characters.
    pub prompt: String,
    /// The number of images to generate. Must be between 1 and 10.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u16>,
    /// The size of the generated images. Must be one of 256x256, 512x512, or 1024x1024.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    /// The format in which the generated images are returned. Must be one of ImageFormat::URL or ImageFormat::B64JSON.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ImageFormat>,
    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

pub type CreateImageEditResponse = CreateImageResponse;

#[derive(Deserialize, Serialize, Debug, Default)]
pub struct CreateImageVariationRequest {
    /// The image to use as the basis for the variation(s). Must be a valid PNG file, less than 4MB, and square.
    pub image: String,
    /// The number of images to generate. Must be between 1 and 10.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u16>,
    /// The size of the generated images. Must be one of 256x256, 512x512, or 1024x1024.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    /// The format in which the generated images are returned. Must be one of ImageFormat::URL or ImageFormat::B64JSON.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ImageFormat>,
    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

pub type CreateImageVariationResponse = CreateImageResponse;

#[derive(Deserialize, Serialize, Debug, Default)]
pub struct CreateEmbeddingsRequest {
    /// ID of the model to use. You can use the List models API to see all of your available models, or see our Model overview for descriptions of them.
    pub model: String,
    /// Input text to get embeddings for, encoded as a string or array of tokens. To get embeddings for multiple inputs in a single request, pass an array of strings or array of token arrays. Each input must not exceed 8192 tokens in length.
    pub input: Vec<String>,
    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct CreateEmbeddingsResponseData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: i64,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct CreateEmbeddingsResponseUsage {
    pub prompt_tokens: i64,
    pub total_tokens: i64,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct CreateEmbeddingsResponse {
    pub object: String,
    pub data: Vec<CreateEmbeddingsResponseData>,
    pub model: String,
    pub usage: CreateEmbeddingsResponseUsage,
}

#[derive(Deserialize, Serialize, Debug, Clone, Copy)]
pub enum CreateTranscriptionResponseFormat {
    #[serde(rename = "json")]
    JSON,
    #[serde(rename = "text")]
    TEXT,
    #[serde(rename = "srt")]
    SRT,
    #[serde(rename = "verbose_json")]
    VERBOSEJSON,
    #[serde(rename = "vtt")]
    VTT,
}

impl Display for CreateTranscriptionResponseFormat {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            CreateTranscriptionResponseFormat::JSON => write!(f, "json"),
            CreateTranscriptionResponseFormat::TEXT => write!(f, "text"),
            CreateTranscriptionResponseFormat::SRT => write!(f, "srt"),
            CreateTranscriptionResponseFormat::VERBOSEJSON => write!(f, "verbose_json"),
            CreateTranscriptionResponseFormat::VTT => write!(f, "vtt"),
        }
    } 
}

#[derive(Deserialize, Serialize, Debug, Default)]
pub struct CreateTranscriptionRequest {
    /// The audio file to transcribe, in one of these formats: mp3, mp4, mpeg, mpga, m4a, wav, or webm.
    pub file: String,
    /// ID of the model to use. Only whisper-1 is currently available.
    pub model: String,
    /// An optional text to guide the model's style or continue a previous audio segment. The prompt should match the audio language.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// The format of the transcript output, in one of these options: json, text, srt, verbose_json, or vtt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<CreateTranscriptionResponseFormat>,
    /// The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. If set to 0, the model will use log probability to automatically increase the temperature until certain thresholds are hit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// The language of the input audio. Supplying the input language in ISO-639-1 format will improve accuracy and latency.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

pub enum CreateTranscriptionResponse {
    Text(CreateTranscriptionResponseText),
    Json(CreateTranscriptionResponseJson),
    Srt(CreateTranscriptionResponseSrt),
    VerboseJson(CreateTranscriptionResponseVerboseJson),
    Vtt(CreateTranscriptionResponseVtt),
}

#[derive(Deserialize, Serialize, Debug)]
pub struct CreateTranscriptionResponseText {
    pub text: String,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct CreateTranscriptionResponseJson {
    pub text: String,
}


#[derive(Deserialize, Serialize, Debug)]
pub struct CreateTranscriptionResponseSrt {
    pub text: String,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct TranscriptionSegment {
    pub id: String,
    pub seek: i32,
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub tokens: Vec<i64>,
    pub temperature: f32,
    pub avg_logprob: f64,
    pub compression_ratio: f64,
    pub no_speech_prob: f64,
    pub transient: bool,
}
    
#[derive(Deserialize, Serialize, Debug)]
pub struct CreateTranscriptionResponseVerboseJson {
    pub task: String,
    pub language: String,
    pub duration: f32,
    pub segments: Vec<TranscriptionSegment>,
    pub text: String,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct CreateTranscriptionResponseVtt {
    pub text: String,
}

#[derive(Deserialize, Serialize, Debug, Default)]
pub struct CreateTranslationRequest {
    /// The audio file to transcribe, in one of these formats: mp3, mp4, mpeg, mpga, m4a, wav, or webm.
    pub file: String,
    /// ID of the model to use. Only whisper-1 is currently available.
    pub model: String,
    /// An optional text to guide the model's style or continue a previous audio segment. The prompt should match the audio language.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// The format of the transcript output, in one of these options: json, text, srt, verbose_json, or vtt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<CreateTranscriptionResponseFormat>,
    /// The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. If set to 0, the model will use log probability to automatically increase the temperature until certain thresholds are hit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

pub type CreateTranslationResponse = CreateTranscriptionResponse;

#[derive(Deserialize, Serialize, Debug)]
pub struct FileInfo {
    pub id: String,
    pub object: String,
    pub bytes: i32,
    pub created_at: i64,
    pub filename: String,
    pub purpose: String,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct ListFilesResponse {
    pub data: Vec<FileInfo>,
    pub object: String,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct UploadFileRequest {
    /// JSON Lines file to be uploaded.
    /// If the purpose is set to "fine-tune", each line is a JSON record with "prompt" and "completion" fields representing your training examples.
    pub file: String,
    /// The name of the file.
    pub filename: String,
    /// The purpose of the file. Can be "fine-tune" or "test".
    pub purpose: String,
}

pub type UploadFileResponse = FileInfo;

#[derive(Deserialize, Serialize, Debug)]
pub struct DeleteFileResponse {
    pub deleted: bool,
    pub id: String,
    pub object: String,
}

pub type RetrieveFileResponse = FileInfo;

#[derive(Deserialize, Serialize, Debug, Default)]
pub struct CreateFineTuneRequest {
    /// The ID of an uploaded file that contains training data.
    /// See upload file for how to upload a file.
    /// Your dataset must be formatted as a JSONL file, where each training example is a JSON object with the keys "prompt" and "completion". 
    /// Additionally, you must upload your file with the purpose fine-tune.
    pub training_file: String,

    /// The ID of an uploaded file that contains validation data.
    /// If you provide this file, the data is used to generate validation metrics periodically during fine-tuning. 
    /// These metrics can be viewed in the fine-tuning results file. Your train and validation data should be mutually exclusive.
    /// Your dataset must be formatted as a JSONL file, where each validation example is a JSON object with the keys "prompt" and "completion". 
    /// Additionally, you must upload your file with the purpose fine-tune.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_file: Option<String>,

    /// The name of the base model to fine-tune. 
    /// You can select one of "ada", "babbage", "curie", "davinci", or a fine-tuned model created after 2022-04-21. 
    /// To learn more about these models
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// The number of epochs to train the model for. An epoch refers to one full cycle through the training dataset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_epochs: Option<i32>,

    /// The batch size to use for training. The batch size is the number of training examples used to train a single forward and backward pass.
    /// By default, the batch size will be dynamically configured to be ~0.2% of the number of examples in the training set, capped at 256 - in general, we've found that larger batch sizes tend to work better for larger datasets.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<i32>,

    /// The learning rate multiplier to use for training. The fine-tuning learning rate is the original learning rate used for pretraining multiplied by this value.
    /// By default, the learning rate multiplier is the 0.05, 0.1, or 0.2 depending on final batch_size (larger learning rates tend to perform better with larger batch sizes). 
    /// We recommend experimenting with values in the range 0.02 to 0.2 to see what produces the best results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub learning_rate_multiplier: Option<f32>,

    /// The weight to use for loss on the prompt tokens. This controls how much the model tries to learn to generate the prompt (as compared to the completion which always has a weight of 1.0), and can add a stabilizing effect to training when completions are short.
    /// If prompts are extremely long (relative to completions), it may make sense to reduce this weight so as to avoid over-prioritizing learning the prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_loss_weight: Option<f32>,

    /// If set, we calculate classification-specific metrics such as accuracy and F-1 score using the validation set at the end of every epoch. 
    /// These metrics can be viewed in the results file.
    /// In order to compute classification metrics, you must provide a validation_file. 
    /// Additionally, you must specify classification_n_classes for multiclass classification or classification_positive_class for binary classification.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compute_classification_metrics: Option<bool>,

    /// The number of classes in a classification task.
    /// This parameter is required for multiclass classification.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub classification_n_classes: Option<i32>,

    /// The positive class in binary classification.
    /// This parameter is needed to generate precision, recall, and F1 metrics when doing binary classification.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub classification_positive_class: Option<String>,

    /// If this is provided, we calculate F-beta scores at the specified beta values. 
    /// The F-beta score is a generalization of F-1 score. This is only used for binary classification.
    /// With a beta of 1 (i.e. the F-1 score), precision and recall are given the same weight. 
    /// A larger beta score puts more weight on recall and less on precision. 
    /// A smaller beta score puts more weight on precision and less on recall.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub classification_betas: Option<Vec<f32>>,

    /// A string of up to 40 characters that will be added to your fine-tuned model name.
    /// For example, a suffix of "custom-model-name" would produce a model name like `ada:ft-your-org:custom-model-name-2022-02-15-04-21-04`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct FineTuneEvent {
    pub object: String,
    pub created_at: i64,
    pub level: String,
    pub message: String,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct FineTuneHyperparams {
    pub batch_size: i32,
    pub learning_rate_multiplier: f32,
    pub prompt_loss_weight: f32,
    pub n_epochs: i32,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct CreateFineTuneResponse {
    pub id: String,
    pub object: String,
    pub model: String,
    pub created_at: i64,
    pub events: Vec<FineTuneEvent>,
    pub fine_tuned_model: Option<String>,
    pub hyperparams: FineTuneHyperparams,
    pub organization_id: String,
    pub result_files: Vec<FileInfo>,
    pub status: String,
    pub validation_files: Vec<FileInfo>,
    pub training_files: Vec<FileInfo>,
    pub updated_at: i64,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct ListFineTunesResponse {
    pub object: String,
    pub data: Vec<CreateFineTuneResponse>,
}

pub type RetrieveFineTuneResponse = CreateFineTuneResponse;

pub type CancelFineTuneResponse = CreateFineTuneResponse;

#[derive(Deserialize, Serialize, Debug)]
pub struct ListFineTuneEventsResponse {
    pub object: String,
    pub data: Vec<FineTuneEvent>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct DeleteFineTuneModelResponse {
    pub id: String,
    pub object: String,
    pub deleted: bool,
}

#[derive(Deserialize, Serialize, Debug, Default)]
pub struct CreateModerationRequest {
    /// The input text to classify
    pub input: Vec<String>,
    /// Two content moderations models are available: text-moderation-stable and text-moderation-latest.
    /// The default is text-moderation-latest which will be automatically upgraded over time. 
    /// This ensures you are always using our most accurate model. 
    /// If you use text-moderation-stable, we will provide advanced notice before updating the model. 
    /// Accuracy of text-moderation-stable may be slightly lower than for text-moderation-latest.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct ModerationCategories {
    pub hate: bool,
    #[serde(rename = "hate/threatening")]
    pub hate_threatening: bool,
    #[serde(rename = "self-harm")]
    pub self_harm: bool,
    pub sexual: bool,
    #[serde(rename = "sexual/minors")]
    pub sexual_minors: bool,
    pub violence: bool,
    #[serde(rename = "violence/graphic")]
    pub violence_graphic: bool,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct ModerationCategoryScores {
    pub hate: f64,
    #[serde(rename = "hate/threatening")]
    pub hate_threatening: f64,
    #[serde(rename = "self-harm")]
    pub self_harm: f64,
    pub sexual: f64,
    #[serde(rename = "sexual/minors")]
    pub sexual_minors: f64,
    pub violence: f64,
    #[serde(rename = "violence/graphic")]
    pub violence_graphic: f64,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct CreateModerationResult {
    pub categories: ModerationCategories,
    pub category_scores: ModerationCategoryScores,
    pub flagged: bool,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct CreateModerationResponse {
    pub id: String,
    pub model: String,
    pub results: Vec<CreateModerationResult>,
}

pub struct OpenAIApi {
    configuration: Configuration,
}

impl OpenAIApi {

    pub fn new(configuration: Configuration) -> Self {
        Self { configuration }
    }

    /// List models
    /// GET https://api.openai.com/v1/models
    /// Lists the currently available models, and provides basic information about each one such as the owner and availability.
    pub async fn list_models(self) -> Result<ListModelsResponse, OpenAIApiError> {

        let client_builder = reqwest::Client::builder();
        let request_builder = self.configuration.apply_to_request(
            client_builder, 
            "/models".to_string(), 
            Method::GET,
        );
        let response = request_builder.send().await
            .map_err(|err| OpenAIApiError::from(err))?;
        if response.status().is_success() {
            response.json::<ListModelsResponse>().await
                .map_err(|err| OpenAIApiError::from(err))
        } else {
            let status = response.status().as_u16() as i32;
            let ret_err = response.json::<ReturnErrorType>().await
                .map_err(|err| OpenAIApiError::from(err))?;
            Err(OpenAIApiError::new(status, ret_err.error))
        }
    }

    /// Retrieve model
    /// GET https://api.openai.com/v1/models/{model}
    /// Retrieves a model instance, providing basic information about the model such as the owner and permissioning.
    pub async fn retrieve_model(self, model: String) -> Result<RetrieveModelResponse, OpenAIApiError> {

        let client_builder = reqwest::Client::builder();
        let request_builder = self.configuration.apply_to_request(
            client_builder, 
            format!("/models/{}", model), 
            Method::GET,
        );
        let response = request_builder.send().await
            .map_err(|err| OpenAIApiError::from(err))?;
        if response.status().is_success() {
            response.json::<RetrieveModelResponse>().await
                .map_err(|err| OpenAIApiError::from(err))
        } else {
            let status = response.status().as_u16() as i32;
            let ret_err = response.json::<ReturnErrorType>().await
                .map_err(|err| OpenAIApiError::from(err))?;
            Err(OpenAIApiError::new(status, ret_err.error))
        }
    }

    /// Create completion
    /// POST https://api.openai.com/v1/completions
    /// Creates a completion for the provided prompt and parameters
    pub async fn create_completion(self, request: CreateCompletionRequest) -> Result<CreateCompletionResponse, OpenAIApiError> {

        let client_builder = reqwest::Client::builder();
        let request_builder = self.configuration.apply_to_request(
            client_builder, 
            "/completions".to_string(), 
            Method::POST,
        );
        let response = request_builder.json(&request).send().await
            .map_err(|err| OpenAIApiError::from(err))?;
        info!("response: {:#?}", response);
        if response.status().is_success() {
            response.json::<CreateCompletionResponse>().await
                .map_err(|err| OpenAIApiError::from(err))
        } else {
            let status = response.status().as_u16() as i32;
            let ret_err = response.json::<ReturnErrorType>().await
                .map_err(|err| OpenAIApiError::from(err))?;
            Err(OpenAIApiError::new(status, ret_err.error))
        }
    }

    ///
    /// Create chat completion
    /// POST https://api.openai.com/v1/chat/completions
    /// Creates a completion for the chat message
    pub async fn create_chat_completion(self, request: CreateChatCompletionRequest) -> Result<CreateChatCompletionResponse, OpenAIApiError> {

        let client_builder = reqwest::Client::builder();
        let request_builder = self.configuration.apply_to_request(
            client_builder, 
            "/chat/completions".to_string(), 
            Method::POST,
        );
        let response = request_builder.json(&request).send().await
            .map_err(|err| OpenAIApiError::from(err))?;
        info!("response: {:#?}", response);
        // println!("response: {:#?}, {}", response, response.status().is_success());
        if response.status().is_success() {
            response.json::<CreateChatCompletionResponse>().await
                .map_err(|err| OpenAIApiError::from(err))
        } else {
            let status = response.status().as_u16() as i32;
            let ret_err = response.json::<ReturnErrorType>().await
                .map_err(|err| OpenAIApiError::from(err))?;
            Err(OpenAIApiError::new(status, ret_err.error))
        }
    }

    /// Create edit
    /// POST https://api.openai.com/v1/edits
    /// Creates a new edit for the provided input, instruction, and parameters.
    pub async fn create_edit(self, request: CreateEditRequest) -> Result<CreateEditResponse, OpenAIApiError> {
        let client_builder = reqwest::Client::builder();
        let request_builder = self.configuration.apply_to_request(
            client_builder, 
            "/edits".to_string(), 
            Method::POST,
        );
        let response = request_builder.json(&request).send().await
            .map_err(|err| OpenAIApiError::from(err))?;
        info!("response: {:#?}", response);
        // println!("response: {:#?}", response.status());
        if response.status().is_success() {
            response.json::<CreateEditResponse>().await
                .map_err(|err| OpenAIApiError::from(err))
        } else {
            let status = response.status().as_u16() as i32;
            let ret_err = response.json::<ReturnErrorType>().await
                .map_err(|err| OpenAIApiError::from(err))?;
            Err(OpenAIApiError::new(status, ret_err.error))
        }
    }

    ///
    /// Create image
    /// POST https://api.openai.com/v1/images/generations
    /// Creates an image given a prompt.
    /// 
    pub async fn create_image(self, request: CreateImageRequest)  -> Result<CreateImageResponse, OpenAIApiError>{
        let client_builder = reqwest::Client::builder();
        let request_builder = self.configuration.apply_to_request(
            client_builder, 
            "/images/generations".to_string(), 
            Method::POST,
        );
        let response = request_builder.json(&request).send().await
            .map_err(|err| OpenAIApiError::from(err))?;
        // println!("body: {:?}", response.unwrap().text().await);
        if response.status().is_success() {
            response.json::<CreateImageResponse>().await
                .map_err(|err| OpenAIApiError::from(err))
        } else {
            let status = response.status().as_u16() as i32;
            let ret_err = response.json::<ReturnErrorType>().await
                .map_err(|err| OpenAIApiError::from(err))?;
            Err(OpenAIApiError::new(status, ret_err.error))
        }
    }

    
    /// Create image editBeta
    /// POST https://api.openai.com/v1/images/edits
    /// Creates an edited or extended image given an original image and a prompt.
    pub async fn create_image_edit(self, request: CreateImageEditRequest) -> Result<CreateImageEditResponse, OpenAIApiError> {
        let client_builder = reqwest::Client::builder();
        let request_builder = self.configuration.apply_to_request(
            client_builder, 
            "/images/edits".to_string(), 
            Method::POST,
        );
        let image_file = fs::read(request.image).unwrap();
        let image_file_part = Part::bytes(image_file)
            .file_name("image.png")
            .mime_str("image/png")
            .unwrap();
        let mut form = reqwest::multipart::Form::new()
        .part("image", image_file_part);
        form = match request.mask {
            Some(mask) => {
                let mask_file = fs::read(mask).unwrap();
                let mask_file_part = Part::bytes(mask_file)
                    .file_name("mask.png")
                    .mime_str("image/png")
                    .unwrap();
                form.part("mask", mask_file_part)
            },
            None => form,
        };
        form = form.text("prompt", request.prompt.clone());
        form = match request.n {
            Some(n) => form.text("n", n.to_string()),
            None => form,
        };
        form = match request.size {
            Some(size) => form.text("size", size),
            None => form,
        };
        form = match request.response_format {
            Some(response_format) => form.text("response_format", response_format.to_string()),
            None => form,
        };
        form = match request.user {
            Some(user) => form.text("user", user),
            None => form,
        };
        let response = request_builder.multipart(form).send().await
            .map_err(|err| OpenAIApiError::from(err))?;
        // println!("response: {:#?}", response);
        if response.status().is_success() {
            response.json::<CreateImageEditResponse>().await
                .map_err(|err| OpenAIApiError::from(err))
        } else {
            let status = response.status().as_u16() as i32;
            let ret_err = response.json::<ReturnErrorType>().await
                .map_err(|err| OpenAIApiError::from(err))?;
            Err(OpenAIApiError::new(status, ret_err.error))
        }
    }
    
    /// Create image variation
    /// POST https://api.openai.com/v1/images/variations
    /// Creates a variation of a given image.
    pub async fn create_image_variation(self, request: CreateImageVariationRequest) -> Result<CreateImageVariationResponse, OpenAIApiError> {
        let client_builder = reqwest::Client::builder();
        let request_builder = self.configuration.apply_to_request(
            client_builder, 
            "/images/variations".to_string(), 
            Method::POST,
        );
        let image_file = fs::read(request.image).unwrap();
        let image_file_part = Part::bytes(image_file)
            .file_name("image.png")
            .mime_str("image/png")
            .unwrap();
        let mut form = reqwest::multipart::Form::new().part("image", image_file_part);
        
        form = match request.n {
            Some(n) => form.text("n", n.to_string()),
            None => form,
        };
        form = match request.size {
            Some(size) => form.text("size", size),
            None => form,
        };
        form = match request.response_format {
            Some(response_format) => form.text("response_format", response_format.to_string()),
            None => form,
        };
        form = match request.user {
            Some(user) => form.text("user", user),
            None => form,
        };
        let response = request_builder.multipart(form).send().await
            .map_err(|err| OpenAIApiError::from(err))?;
        // println!("response: {:#?}", response);
        if response.status().is_success() {
            response.json::<CreateImageVariationResponse>().await
                .map_err(|err| OpenAIApiError::from(err))
        } else {
            let status = response.status().as_u16() as i32;
            let ret_err = response.json::<ReturnErrorType>().await
                .map_err(|err| OpenAIApiError::from(err))?;
            Err(OpenAIApiError::new(status, ret_err.error))
        }
    }

    /// Create embeddings
    /// POST https://api.openai.com/v1/embeddings
    /// Creates an embedding vector representing the input text.
    pub async fn create_embeddings(self, request: CreateEmbeddingsRequest) -> Result<CreateEmbeddingsResponse, OpenAIApiError> {
        let client_builder = reqwest::Client::builder();
        let request_builder = self.configuration.apply_to_request(
            client_builder, 
            "/embeddings".to_string(), 
            Method::POST,
        );
        let response = request_builder.json(&request).send().await
            .map_err(|err| OpenAIApiError::from(err))?;
        info!("response: {:#?}", response);
        if response.status().is_success() {
            response.json::<CreateEmbeddingsResponse>().await
                .map_err(|err| OpenAIApiError::from(err))
        } else {
            let status = response.status().as_u16() as i32;
            let ret_err = response.json::<ReturnErrorType>().await
                .map_err(|err| OpenAIApiError::from(err))?;
            Err(OpenAIApiError::new(status, ret_err.error))
        }
    }

    /// Create transcription
    /// POST https://api.openai.com/v1/audio/transcriptions
    /// Transcribes audio into the input language.
    pub async fn create_transcription(self, request: CreateTranscriptionRequest) -> Result<CreateTranscriptionResponse, OpenAIApiError> {
        let client_builder = reqwest::Client::builder();
        let request_builder = self.configuration.apply_to_request(
            client_builder, 
            "/audio/transcriptions".to_string(), 
            Method::POST,
        );
        let parts: Vec<&str> = request.file.split('.').collect();
        let suffix = parts[parts.len() - 1];
        let mime_type = Self::get_mime_type_from_suffix(suffix.to_string())?;
        let audio_file = fs::read(request.file.clone()).unwrap();
        let audio_file_part = Part::bytes(audio_file)
            .file_name(format!("audio.{}", suffix))
            .mime_str(mime_type.as_str())
            .unwrap();
        let mut form = reqwest::multipart::Form::new().part("file", audio_file_part)
            .text("model", request.model);
        form = match request.prompt {
            Some(prompt) => form.text("prompt", prompt),
            None => form,
        };
        form = match request.response_format {
            Some(response_format) => form.text("response_format", response_format.to_string()),
            None => form,
        };
        form = match request.temperature {
            Some(temperature) => form.text("temperature", temperature.to_string()),
            None => form,
        };
        form = match request.language {
            Some(language) => form.text("language", language),
            None => form,
        };
        info!("request form: {:#?}", form);
        let response = request_builder.multipart(form).send().await
            .map_err(|err| OpenAIApiError::from(err))?;
        println!("response: {:#?}", response);
        let rf = request.response_format.clone();
        if response.status().is_success() {
            match rf {
                Some(response_format) => match response_format {
                    CreateTranscriptionResponseFormat::TEXT => {
                        let text = response.text().await
                            .map_err(|err| OpenAIApiError::from(err))?;
                        let response = CreateTranscriptionResponseText {
                            text,
                        };
                        Ok(CreateTranscriptionResponse::Text(response))
                    },
                    CreateTranscriptionResponseFormat::JSON => {
                        let response = response.json::<CreateTranscriptionResponseJson>().await
                            .map_err(|err| OpenAIApiError::from(err)).unwrap();
                        Ok(CreateTranscriptionResponse::Json(response))
                    },
                    CreateTranscriptionResponseFormat::SRT => {
                        let text = response.text().await
                            .map_err(|err| OpenAIApiError::from(err))?;
                        let response = CreateTranscriptionResponseSrt {
                            text,
                        };
                        Ok(CreateTranscriptionResponse::Srt(response))
                    },
                    CreateTranscriptionResponseFormat::VTT => {
                        let text = response.text().await
                            .map_err(|err| OpenAIApiError::from(err))?;
                        let response = CreateTranscriptionResponseVtt {
                            text,
                        };
                        Ok(CreateTranscriptionResponse::Vtt(response))
                    },
                    CreateTranscriptionResponseFormat::VERBOSEJSON => {
                        let response = response.json::<CreateTranscriptionResponseVerboseJson>().await
                            .map_err(|err| OpenAIApiError::from(err))?;
                        Ok(CreateTranscriptionResponse::VerboseJson(response))
                    },
                },
                None => {
                    let response = response.json::<CreateTranscriptionResponseJson>().await
                        .map_err(|err| OpenAIApiError::from(err))?;
                    Ok(CreateTranscriptionResponse::Json(response))
                },
            }
        } else {
            let status = response.status().as_u16() as i32;
            let ret_err = response.json::<ReturnErrorType>().await
                .map_err(|err| OpenAIApiError::from(err))?;
            Err(OpenAIApiError::new(status, ret_err.error))
        }
        
    }

    /// Create translation
    /// POST https://api.openai.com/v1/audio/translations
    /// Translates audio into into English.
    pub async fn create_translation(self, request: CreateTranslationRequest) -> Result<CreateTranslationResponse, OpenAIApiError> {
        let client_builder = reqwest::Client::builder();
        let request_builder = self.configuration.apply_to_request(
            client_builder, 
            "/audio/translations".to_string(), 
            Method::POST,
        );
        let parts: Vec<&str> = request.file.split('.').collect();
        let suffix = parts[parts.len() - 1];
        let mime_type = Self::get_mime_type_from_suffix(suffix.to_string()).unwrap();
        let audio_file = fs::read(request.file.clone()).unwrap();
        let audio_file_part = Part::bytes(audio_file)
            .file_name(format!("audio.{}", suffix))
            .mime_str(mime_type.as_str())
            .unwrap();
        let mut form = reqwest::multipart::Form::new().part("file", audio_file_part)
            .text("model", request.model);
        form = match request.prompt {
            Some(prompt) => form.text("prompt", prompt),
            None => form,
        };
        form = match request.response_format {
            Some(response_format) => form.text("response_format", response_format.to_string()),
            None => form,
        };
        form = match request.temperature {
            Some(temperature) => form.text("temperature", temperature.to_string()),
            None => form,
        };
        let response = request_builder.multipart(form).send().await
            .map_err(|err| OpenAIApiError::from(err))?;
        info!("response: {:#?}", response);
        let rf = request.response_format.clone();
        if response.status().is_success() {
            match rf {
                Some(response_format) => match response_format {
                    CreateTranscriptionResponseFormat::TEXT => {
                        let text = response.text().await
                            .map_err(|err| OpenAIApiError::from(err))?;
                        let response = CreateTranscriptionResponseText {
                            text,
                        };
                        Ok(CreateTranslationResponse::Text(response))
                    },
                    CreateTranscriptionResponseFormat::JSON => {
                        let response = response.json::<CreateTranscriptionResponseJson>().await
                            .map_err(|err| OpenAIApiError::from(err))?;
                        Ok(CreateTranslationResponse::Json(response))
                    },
                    CreateTranscriptionResponseFormat::SRT => {
                        let text = response.text().await
                            .map_err(|err| OpenAIApiError::from(err))?;
                        let response = CreateTranscriptionResponseSrt {
                            text,
                        };
                        Ok(CreateTranslationResponse::Srt(response))
                    },
                    CreateTranscriptionResponseFormat::VTT => {
                        let text = response.text().await
                            .map_err(|err| OpenAIApiError::from(err))?;
                        let response = CreateTranscriptionResponseVtt {
                            text,
                        };
                        Ok(CreateTranslationResponse::Vtt(response))
                    },
                    CreateTranscriptionResponseFormat::VERBOSEJSON => {
                        let response = response.json::<CreateTranscriptionResponseVerboseJson>().await
                            .map_err(|err| OpenAIApiError::from(err))?;
                        Ok(CreateTranslationResponse::VerboseJson(response))
                    },
                },
                None => {
                    let response = response.json::<CreateTranscriptionResponseJson>().await
                        .map_err(|err| OpenAIApiError::from(err))?;
                    Ok(CreateTranslationResponse::Json(response))
                },
            }
        } else {
            let status = response.status().as_u16() as i32;
            let ret_err = response.json::<ReturnErrorType>().await
                .map_err(|err| OpenAIApiError::from(err))?;
            Err(OpenAIApiError::new(status, ret_err.error))
        }
        
        
    }

    /// List files
    /// GET https://api.openai.com/v1/files
    /// Returns a list of files that belong to the user's organization.
    pub async fn list_files(self) -> Result<ListFilesResponse, OpenAIApiError> {
        let client_builder = reqwest::Client::builder();
        let request_builder = self.configuration.apply_to_request(
            client_builder, 
            "/files".to_string(), 
            Method::GET,
        );
        let response = request_builder.send().await
            .map_err(|err| OpenAIApiError::from(err))?;
        if response.status().is_success() {
            response.json::<ListFilesResponse>().await
                .map_err(|err| OpenAIApiError::from(err))
        } else {
            let status = response.status().as_u16() as i32;
            let ret_err = response.json::<ReturnErrorType>().await
                .map_err(|err| OpenAIApiError::from(err))?;
            Err(OpenAIApiError::new(status, ret_err.error))
        }
    }

    /// Upload file
    /// POST https://api.openai.com/v1/files
    /// Upload a file that contains document(s) to be used across various endpoints/features. Currently, the size of all the files uploaded by one organization can be up to 1 GB. Please contact us if you need to increase the storage limit.
    pub async fn upload_file(self, request: UploadFileRequest) -> Result<UploadFileResponse, OpenAIApiError> {
        let client_builder = reqwest::Client::builder();
        let request_builder = self.configuration.apply_to_request(
            client_builder, 
            "/files".to_string(), 
            Method::POST,
        );
        let file = fs::read(request.file.clone()).unwrap();
        let file_part = Part::bytes(file)
            .file_name(request.file.clone())
            .mime_str(mime::APPLICATION_JSON.to_string().as_str())
            .unwrap();
        let form = reqwest::multipart::Form::new().part("file", file_part)
            .text("purpose", request.purpose);
        let response = request_builder.multipart(form).send().await
            .map_err(|err| OpenAIApiError::from(err))?;
        if response.status().is_success() {
            response.json::<UploadFileResponse>().await
                .map_err(|err| OpenAIApiError::from(err))
        } else {
            let status = response.status().as_u16() as i32;
            let ret_err = response.json::<ReturnErrorType>().await
                .map_err(|err| OpenAIApiError::from(err))?;
            Err(OpenAIApiError::new(status, ret_err.error))
        }
    }

    /// Delete file
    /// DELETE https://api.openai.com/v1/files/{file_id}
    /// Delete a file.
    pub async fn delete_file(self, file_id: String) -> Result<DeleteFileResponse, OpenAIApiError> {
        let client_builder = reqwest::Client::builder();
        let request_builder = self.configuration.apply_to_request(
            client_builder, 
            format!("/files/{}", file_id), 
            Method::DELETE,
        );
        let response = request_builder.send().await
            .map_err(|err| OpenAIApiError::from(err))?;
        if response.status().is_success() {
            response.json::<DeleteFileResponse>().await
                .map_err(|err| OpenAIApiError::from(err))
        } else {
            let status = response.status().as_u16() as i32;
            let ret_err = response.json::<ReturnErrorType>().await
                .map_err(|err| OpenAIApiError::from(err))?;
            Err(OpenAIApiError::new(status, ret_err.error))
        }
    }

    /// Retrieve file
    /// GET https://api.openai.com/v1/files/{file_id}
    /// Returns information about a specific file.
    pub async fn retrieve_file(self, file_id: String) -> Result<RetrieveFileResponse, OpenAIApiError> {
        let client_builder = reqwest::Client::builder();
        let request_builder = self.configuration.apply_to_request(
            client_builder, 
            format!("/files/{}", file_id), 
            Method::GET,
        );
        let response = request_builder.send().await
            .map_err(|err| OpenAIApiError::from(err))?;
        if response.status().is_success() {
            response.json::<RetrieveFileResponse>().await
                .map_err(|err| OpenAIApiError::from(err))
        } else {
            let status = response.status().as_u16() as i32;
            let ret_err = response.json::<ReturnErrorType>().await
                .map_err(|err| OpenAIApiError::from(err))?;
            Err(OpenAIApiError::new(status, ret_err.error))
        }
    }

    /// Retrieve file content
    /// GET https://api.openai.com/v1/files/{file_id}/content
    /// Returns the contents of the specified file
    pub async fn retrieve_file_content(self, file_id: String) -> Result<String, OpenAIApiError> {
        let client_builder = reqwest::Client::builder();
        let request_builder = self.configuration.apply_to_request(
            client_builder, 
            format!("/files/{}/content", file_id), 
            Method::GET,
        );
        let response = request_builder.send().await
            .map_err(|err| OpenAIApiError::from(err))?;
        if response.status().is_success() {
            response.text().await
                .map_err(|err| OpenAIApiError::from(err))
        } else {
            let status = response.status().as_u16() as i32;
            let ret_err = response.json::<ReturnErrorType>().await
                .map_err(|err| OpenAIApiError::from(err))?;
            Err(OpenAIApiError::new(status, ret_err.error))
        }
    }

    /// Create fine-tune
    /// POST https://api.openai.com/v1/fine-tunes
    /// Creates a job that fine-tunes a specified model from a given dataset.
    /// Response includes details of the enqueued job including job status and the name of the fine-tuned models once complete.
    pub async fn create_fine_tune(self, request: CreateFineTuneRequest) -> Result<CreateFineTuneResponse, OpenAIApiError> {
        let client_builder = reqwest::Client::builder();
        let request_builder = self.configuration.apply_to_request(
            client_builder, 
            "/fine-tunes".to_string(), 
            Method::POST,
        );
        let response = request_builder.json(&request).send().await
            .map_err(|err| OpenAIApiError::from(err))?;
        if response.status().is_success() {
            response.json::<CreateFineTuneResponse>().await
                .map_err(|err| OpenAIApiError::from(err))
        } else {
            let status = response.status().as_u16() as i32;
            let ret_err = response.json::<ReturnErrorType>().await
                .map_err(|err| OpenAIApiError::from(err))?;
            Err(OpenAIApiError::new(status, ret_err.error))
        }
    }

    /// List fine-tunes
    /// GET https://api.openai.com/v1/fine-tunes
    /// List your organization's fine-tuning jobs
    pub async fn list_fine_tunes(self) -> Result<ListFineTunesResponse, OpenAIApiError> {
        let client_builder = reqwest::Client::builder();
        let request_builder = self.configuration.apply_to_request(
            client_builder, 
            "/fine-tunes".to_string(), 
            Method::GET,
        );
        let response = request_builder.send().await
            .map_err(|err| OpenAIApiError::from(err))?;
        if response.status().is_success() {
            response.json::<ListFineTunesResponse>().await
                .map_err(|err| OpenAIApiError::from(err))
        } else {
            let status = response.status().as_u16() as i32;
            let ret_err = response.json::<ReturnErrorType>().await
                .map_err(|err| OpenAIApiError::from(err))?;
            Err(OpenAIApiError::new(status, ret_err.error))
        }
    }

    /// Retrieve fine-tune
    /// GET https://api.openai.com/v1/fine-tunes/{fine_tune_id}
    /// Gets info about the fine-tune job.
    pub async fn retrieve_fine_tune(self, fine_tune_id: String) -> Result<RetrieveFineTuneResponse, OpenAIApiError> {
        let client_builder = reqwest::Client::builder();
        let request_builder = self.configuration.apply_to_request(
            client_builder, 
            format!("/fine-tunes/{}", fine_tune_id), 
            Method::GET,
        );
        let response = request_builder.send().await
            .map_err(|err| OpenAIApiError::from(err))?;
        if response.status().is_success() {
            response.json::<RetrieveFineTuneResponse>().await
                .map_err(|err| OpenAIApiError::from(err))
        } else {
            let status = response.status().as_u16() as i32;
            let ret_err = response.json::<ReturnErrorType>().await
                .map_err(|err| OpenAIApiError::from(err))?;
            Err(OpenAIApiError::new(status, ret_err.error))
        }
    }

    /// Cancel fine-tune
    /// POST https://api.openai.com/v1/fine-tunes/{fine_tune_id}/cancel
    /// Immediately cancel a fine-tune job.
    pub async fn cancel_fine_tune(self, fine_tune_id: String) -> Result<CancelFineTuneResponse, OpenAIApiError> {
        let client_builder = reqwest::Client::builder();
        let request_builder = self.configuration.apply_to_request(
            client_builder, 
            format!("/fine-tunes/{}/cancel", fine_tune_id), 
            Method::POST,
        );
        let response = request_builder.send().await
            .map_err(|err| OpenAIApiError::from(err))?;
        if response.status().is_success() {
            response.json::<CancelFineTuneResponse>().await
                .map_err(|err| OpenAIApiError::from(err))
        } else {
            let status = response.status().as_u16() as i32;
            let ret_err = response.json::<ReturnErrorType>().await
                .map_err(|err| OpenAIApiError::from(err))?;
            Err(OpenAIApiError::new(status, ret_err.error))
        }
    }

    /// List fine-tune events
    /// GET https://api.openai.com/v1/fine-tunes/{fine_tune_id}/events
    /// Get fine-grained status updates for a fine-tune job.
    pub async fn list_fine_tune_events(self, fine_tune_id: String) -> Result<ListFineTuneEventsResponse, OpenAIApiError> {
        let client_builder = reqwest::Client::builder();
        let request_builder = self.configuration.apply_to_request(
            client_builder, 
            format!("/fine-tunes/{}/events", fine_tune_id), 
            Method::GET,
        );
        let response = request_builder.send().await
            .map_err(|err| OpenAIApiError::from(err))?;
        if response.status().is_success() {
            response.json::<ListFineTuneEventsResponse>().await
                .map_err(|err| OpenAIApiError::from(err))
        } else {
            let status = response.status().as_u16() as i32;
            let ret_err = response.json::<ReturnErrorType>().await
                .map_err(|err| OpenAIApiError::from(err))?;
            Err(OpenAIApiError::new(status, ret_err.error))
        }
    }

    /// Delete fine-tune model
    /// DELETE https://api.openai.com/v1/models/{model}
    /// Delete a fine-tuned model. You must have the Owner role in your organization.
    pub async fn delete_fine_tune_model(self, model: String) -> Result<DeleteFineTuneModelResponse, OpenAIApiError> {
        let client_builder = reqwest::Client::builder();
        let request_builder = self.configuration.apply_to_request(
            client_builder, 
            format!("/models/{}", model), 
            Method::DELETE,
        );
        let response = request_builder.send().await
            .map_err(|err| OpenAIApiError::from(err))?;
        if response.status().is_success() {
            response.json::<DeleteFineTuneModelResponse>().await
                .map_err(|err| OpenAIApiError::from(err))
        } else {
            let status = response.status().as_u16() as i32;
            let ret_err = response.json::<ReturnErrorType>().await
                .map_err(|err| OpenAIApiError::from(err))?;
            Err(OpenAIApiError::new(status, ret_err.error))
        }
    }

    /// Create moderation
    /// POST https://api.openai.com/v1/moderations
    /// Classifies if text violates OpenAI's Content Policy
    pub async fn create_moderation(self, request: CreateModerationRequest) -> Result<CreateModerationResponse, OpenAIApiError> {
        let client_builder = reqwest::Client::builder();
        let request_builder = self.configuration.apply_to_request(
            client_builder, 
            "/moderations".to_string(), 
            Method::POST,
        );
        let response = request_builder.json(&request).send().await
            .map_err(|err| OpenAIApiError::from(err))?;
        if response.status().is_success() {
            response.json::<CreateModerationResponse>().await
                .map_err(|err| OpenAIApiError::from(err))
        } else {
            let status = response.status().as_u16() as i32;
            let ret_err = response.json::<ReturnErrorType>().await
                .map_err(|err| OpenAIApiError::from(err))?;
            Err(OpenAIApiError::new(status, ret_err.error))
        }
    }

    fn get_mime_type_from_suffix(suffix: String) -> Result<String, OpenAIApiError> {
        match suffix.as_str() {
            "json" => Ok(mime::APPLICATION_JSON.to_string()),
            "txt" => Ok(mime::TEXT_PLAIN.to_string()),
            "html" => Ok(mime::TEXT_HTML.to_string()),
            "pdf" => Ok(mime::APPLICATION_PDF.to_string()),
            "png" => Ok(mime::IMAGE_PNG.to_string()),
            "jpg" => Ok(mime::IMAGE_JPEG.to_string()),
            "jpeg" => Ok(mime::IMAGE_JPEG.to_string()),
            "gif" => Ok(mime::IMAGE_GIF.to_string()),
            "svg" => Ok(mime::IMAGE_SVG.to_string()),
            "m4a" => Ok("audio/m4a".to_string()),
            "mp3" => Ok("audio/mp3".to_string()),
            "wav" => Ok("audio/wav".to_string()),
            "flac" => Ok("audio/flac".to_string()),
            "mp4" => Ok("video/mp4".to_string()),
            "mpeg" => Ok("video/mpeg".to_string()),
            "mpga" => Ok("audio/mpeg".to_string()),
            "webm" => Ok("video/webm".to_string()),
            _ => {
                let e = ErrorInfo {
                    message: format!("Unsupported file type: {}", suffix),
                    code: None,
                    message_type: "unsupported_file_type".to_string(),
                    param: None,
                };
                Err(OpenAIApiError::new(400, e))
            },
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configuration::Configuration;
    use dotenv::vars;

    #[tokio::test]
    async fn test_list_models() {

        let api_key = vars().find(|(key, _)| key == "API_KEY").unwrap_or(("API_KEY".to_string(),"".to_string())).1;

        let configuration = Configuration::new_personal(api_key)
            .proxy("http://127.0.0.1:7890".to_string());

        let openai_api = OpenAIApi::new(configuration);
        let response = openai_api.list_models().await.unwrap();
        assert_eq!(response.object, "list");
    }

    #[tokio::test]
    async fn test_retrieve_model() {
        let api_key = vars().find(|(key, _)| key == "API_KEY").unwrap_or(("API_KEY".to_string(),"".to_string())).1;

        let configuration = Configuration::new_personal(api_key)
            .proxy("http://127.0.0.1:7890".to_string());

        let openai_api = OpenAIApi::new(configuration);
        let response = openai_api.retrieve_model("davinci".to_string()).await.unwrap();
        assert_eq!(response.object, "model");
    }

    #[tokio::test]
    async fn test_create_completion() {
        let api_key = vars().find(|(key, _)| key == "API_KEY").unwrap_or(("API_KEY".to_string(),"".to_string())).1;

        let configuration = Configuration::new_personal(api_key)
            .proxy("http://127.0.0.1:7890".to_string());

        let openai_api = OpenAIApi::new(configuration);
        let request = CreateCompletionRequest {
            model: "text-davinci-003".to_string(),
            prompt: Some(vec!["Once upon a time".to_string()]),
            max_tokens: Some(7),
            temperature: Some(0.7),
            ..Default::default()
        };
        
        // println!("request: {:#?}", serde_json::to_string(&request).unwrap());
        let response = openai_api.create_completion(request).await.unwrap();
        assert_eq!(response.object, "text_completion");
    }

    #[tokio::test]
    async fn test_create_chat_completion() {
        let api_key = vars().find(|(key, _)| key == "API_KEY").unwrap_or(("API_KEY".to_string(),"".to_string())).1;

        let configuration = Configuration::new_personal(api_key)
            .proxy("http://127.0.0.1:7890".to_string());

        let openai_api = OpenAIApi::new(configuration);
        let request = CreateChatCompletionRequest {
            model: "gpt-3.5-turbo".to_string(),
            messages: vec![ChatFormat{role: "user".to_string(), content: "tell me a story".to_string()}],
            ..Default::default()
        };
        // println!("request: {:#?}", serde_json::to_string(&request).unwrap());
        let response = openai_api.create_chat_completion(request).await.unwrap();
        assert_eq!(response.object, "chat.completion");
    }

    #[tokio::test]
    async fn test_create_edit() {
        let api_key = vars().find(|(key, _)| key == "API_KEY").unwrap_or(("API_KEY".to_string(),"".to_string())).1;

        let configuration = Configuration::new_personal(api_key)
            .proxy("http://127.0.0.1:7890".to_string());

        let openai_api = OpenAIApi::new(configuration);
        let request = CreateEditRequest {
            model: "text-davinci-edit-001".to_string(),
            input: Some("What day of the wek is it?".to_string()),
            instruction: "Fix the spelling mistakes".to_string(),
            ..Default::default()
        };
        // println!("request: {:#?}", serde_json::to_string(&request).unwrap());
        let response = openai_api.create_edit(request).await.unwrap();
        assert_eq!(response.object, "edit");
    }

    #[tokio::test]
    async fn test_create_image() {
        let api_key = vars().find(|(key, _)| key == "API_KEY").unwrap_or(("API_KEY".to_string(),"".to_string())).1;

        let configuration = Configuration::new_personal(api_key)
            .proxy("http://127.0.0.1:7890".to_string());

        let openai_api = OpenAIApi::new(configuration);
        let request = CreateImageRequest {
            prompt: "A photo of a dog".to_string(),
            n: Some(1),
            size: Some("512x512".to_string()),
            response_format: Some(ImageFormat::URL),
            ..Default::default()
        };
        println!("request: {:#?}", serde_json::to_string(&request).unwrap());
        let response = openai_api.create_image(request).await.unwrap();
        
        assert_eq!(response.data.len(), 1);
        match response.data[0].clone() {
            CreateImageResponseData::Url(url) => {
                assert!(url.starts_with("https://"));
            },
            _ => {
                assert!(false, "error response format");
            }
        }
    }

    #[tokio::test]
    async fn test_create_transcription() {
        let api_key = vars().find(|(key, _)| key == "API_KEY").unwrap_or(("API_KEY".to_string(),"".to_string())).1;

        let configuration = Configuration::new_personal(api_key)
            .proxy("http://127.0.0.1:7890".to_string());

        let openai_api = OpenAIApi::new(configuration);
        let request = CreateTranscriptionRequest {
            file: "./misc/test_audio.m4a".to_string(),
            model: "whisper-1".to_string(),
            response_format: Some(CreateTranscriptionResponseFormat::JSON),
            ..Default::default()
        };
        println!("request: {:#?}", serde_json::to_string(&request).unwrap());
        let response = openai_api.create_transcription(request).await.unwrap();
        match response {
            CreateTranscriptionResponse::Json(content) => {
                assert_eq!(content.text, "你好你好");
            },
            _ => {
                assert!(false);
            }
        };
    }

    #[tokio::test]
    async fn test_create_translation() {
        let api_key = vars().find(|(key, _)| key == "API_KEY").unwrap_or(("API_KEY".to_string(),"".to_string())).1;

        let configuration = Configuration::new_personal(api_key)
            .proxy("http://127.0.0.1:7890".to_string());

        let openai_api = OpenAIApi::new(configuration);
        let request = CreateTranslationRequest {
            file: "./misc/test_audio.m4a".to_string(),
            model: "whisper-1".to_string(),
            response_format: Some(CreateTranscriptionResponseFormat::JSON),
            ..Default::default()
        };
        println!("request: {:#?}", serde_json::to_string(&request).unwrap());
        let response = openai_api.create_translation(request).await.unwrap();
        match response {
            CreateTranslationResponse::Json(content) => {
                assert_eq!(content.text, "Ni hao, ni hao.");
            },
            _ => {
                assert!(false);
            }
        };
    }

    #[tokio::test]
    async fn test_list_files() {
        let api_key = vars().find(|(key, _)| key == "API_KEY").unwrap_or(("API_KEY".to_string(),"".to_string())).1;

        let configuration = Configuration::new_personal(api_key)
            .proxy("http://127.0.0.1:7890".to_string());

        let openai_api = OpenAIApi::new(configuration);
        let response = openai_api.list_files().await.unwrap();
        assert_eq!(response.object, "list");
    }

    #[tokio::test]
    async fn test_list_fine_tunes() {
        let api_key = vars().find(|(key, _)| key == "API_KEY").unwrap_or(("API_KEY".to_string(),"".to_string())).1;

        let configuration = Configuration::new_personal(api_key)
            .proxy("http://127.0.0.1:7890".to_string());

        let openai_api = OpenAIApi::new(configuration);
        let response = openai_api.list_fine_tunes().await.unwrap();
        assert_eq!(response.object, "list");
    }

    #[tokio::test]
    async fn test_create_moderation() {
        let api_key = vars().find(|(key, _)| key == "API_KEY").unwrap_or(("API_KEY".to_string(),"".to_string())).1;

        let configuration = Configuration::new_personal(api_key)
            .proxy("http://127.0.0.1:7890".to_string());

        let openai_api = OpenAIApi::new(configuration);
        let response = openai_api.create_moderation(CreateModerationRequest {
            input: vec!["I want to kill them.".to_string()],
            ..Default::default()
        }).await.unwrap();
        // println!("response: {:#?}", response);
        assert!(response.results[0].categories.violence);
    }

}
    




