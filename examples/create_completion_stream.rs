use dotenv::vars;
use ru_openai::{configuration::Configuration, api::*};
use futures::StreamExt;

#[tokio::main]
async fn main() {
    // Load API key from .env file
    let api_key = vars().find(|(key, _)| key == "API_KEY").unwrap_or(("API_KEY".to_string(),"".to_string())).1;
    // Create a configuration object, with a proxy if you need one. If not, just remove the proxy method.
    let configuration = Configuration::new_personal(api_key)
        .proxy("http://127.0.0.1:7890".to_string());

    let openai_api = OpenAIApi::new(configuration);
    // Create a request object
    let request = CreateCompletionRequest {
        model: "text-davinci-003".to_string(),
        prompt: Some(vec!["Tell me a bedtime story about little swan".to_string()]),
        max_tokens: Some(1024_u64),
        temperature: Some(0.7),
        ..Default::default()
    };
    // Call the API: `create_completion`
    let mut stream = openai_api.create_completion_stream(request).await.unwrap();
    // println!("Answer: {}", response.choices[0].message.content);
    while let Some(response) = stream.next().await {
        match response {
            Ok(create_complete_response) => create_complete_response.choices.iter().for_each(|c| {
                print!("{}", c.text)
            }),
            Err(e) => println!("Error: {:?}", e)
        }
    }
}