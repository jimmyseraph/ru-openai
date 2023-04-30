use dotenv::vars;
use ru_openai::{configuration::Configuration, api::*};
use futures::StreamExt;

#[tokio::main]
async fn main() {
    let api_key = vars().find(|(key, _)| key == "API_KEY").unwrap_or(("API_KEY".to_string(),"".to_string())).1;

    let configuration = Configuration::new_personal(api_key)
        .proxy("http://127.0.0.1:7890".to_string());

    let openai_api = OpenAIApi::new(configuration);
    let content = "How to design a suite of test cases to verify a website.".to_string();
    let request = CreateChatCompletionRequest {
        model: "gpt-3.5-turbo".to_string(),
        messages: vec![ChatFormat{role: "user".to_string(), content: content.clone()}],
        ..Default::default()
    };
    println!("Question: {}", content);
    let mut stream = openai_api.create_chat_completion_stream(request).await.unwrap();
    // println!("Answer: {}", response.choices[0].message.content);
    while let Some(response) = stream.next().await {
        match response {
            Ok(create_chat_complete_stream_response) => create_chat_complete_stream_response.choices.iter().for_each(|c| {
                if let Some(ref content) = c.delta.content {
                    print!("{}", content);
                }
            }),
            Err(e) => println!("Error: {:?}", e)
        }
    }

}