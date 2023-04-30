use std::pin::Pin;
use futures::{stream::StreamExt, Stream};
use reqwest::StatusCode;
use serde_derive::{Deserialize, Serialize};
use reqwest_eventsource::{EventSource, Event};
use serde::de::DeserializeOwned;

pub mod api;
pub mod configuration;

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

pub(crate) async fn stream<O>(
    mut event_source: EventSource,
) -> Pin<Box<dyn Stream<Item = Result<O, OpenAIApiError>> + Send>>
where
    O: DeserializeOwned + std::marker::Send + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

    tokio::spawn(async move {
        while let Some(ev) = event_source.next().await {
            match ev {
                Err(e) => {
                    // println!("{:?}", e);
                    if let Err(_e) = tx.send(Err(OpenAIApiError::new(
                        StatusCode::INTERNAL_SERVER_ERROR.as_u16() as i32,
                        ErrorInfo {
                            message: e.to_string(),
                            message_type: "request error".to_string(),
                            param: None,
                            code: None,
                        },
                    ))) {
                        // rx dropped
                        break;
                    }
                }
                Ok(event) => match event {
                    Event::Message(message) => {
                        // println!("{:?}", message);
                        if message.data == "[DONE]" {
                            break;
                        }

                        let response = match serde_json::from_str::<O>(&message.data) {
                            Err(e) => {
                                // Err(map_deserialization_error(e, &message.data.as_bytes()))

                                Err(OpenAIApiError::new(
                                    StatusCode::INTERNAL_SERVER_ERROR.as_u16() as i32,
                                    ErrorInfo {
                                        message: e.to_string(),
                                        message_type: "deserialization error".to_string(),
                                        param: None,
                                        code: None,
                                    },
                                ))
                            }
                            Ok(output) => Ok(output),
                        };

                        if let Err(_e) = tx.send(response) {
                            // rx dropped
                            break;
                        }
                    }
                    Event::Open => continue,
                },
            }
        }

        event_source.close();
    });

    Box::pin(tokio_stream::wrappers::UnboundedReceiverStream::new(rx))
}