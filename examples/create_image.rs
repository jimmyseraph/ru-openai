use dotenv::vars;
use ru_openai::{configuration::Configuration, api::*};
#[tokio::main]
async fn main() {
    let api_key = vars().find(|(key, _)| key == "API_KEY").unwrap_or(("API_KEY".to_string(),"".to_string())).1;

    let configuration = Configuration::new_personal(api_key)
        .proxy("http://127.0.0.1:7890".to_string());

    let openai_api = OpenAIApi::new(configuration);
    let request = CreateImageRequest {
        prompt: "一个在海边的性感年轻的亚裔姑娘".to_string(),
        n: Some(3),
        size: Some("512x512".to_string()),
        response_format: Some(ImageFormat::URL),
        ..Default::default()
    };
    let response = openai_api.create_image(request).await.unwrap();
    println!("Answer: {:#?}", response);
}