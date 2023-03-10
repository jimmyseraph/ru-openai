# Ru-OpenAi
Rust library for the OpenAI API. In fact, maybe this is the best crate in rust.

## About OpenAiAPI
You can learn everything you want to know from [OpenAI API reference].

## Example
An example about how to call `create completion` API.

Make sure you add the dependence on your Cargo.toml:
```toml
[dependencies]
ru-openai = "0.1.2"
```
Then create a file named `.env` in your project root directory. Input your own openai `api-key` in it:
```
API_KEY = your-api-key
```

Then you can call API like this:
```rust
use dotenv::vars;
use ru_openai::{configuration::Configuration, api::*};
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
        prompt: Some(vec!["Once upon a time".to_string()]),
        max_tokens: Some(7),
        temperature: Some(0.7),
        ..Default::default()
    };
    // Call the API: `create_completion`
    let response = openai_api.create_completion(request).await.unwrap();
    println!("{:#?}", response);
}
```

[OpenAI API reference]: https://platform.openai.com/docs/api-reference