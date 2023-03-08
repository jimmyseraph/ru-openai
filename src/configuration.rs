use std::time::Duration;


const API_BASE_PATH: &str = "https://api.openai.com/v1";

const USER_AGENT_VALUE: &str = "openai-rust/0.1.0";

#[derive(Debug)]
pub struct Configuration {
    pub api_key: Option<String>,
    pub organization: Option<String>,
    pub base_path: String,
    pub proxy: Option<String>,
    pub timeout: Option<u64>,
}

impl Configuration {
    pub fn new(
        api_key: Option<String>, 
        organization: Option<String>,
    ) -> Self {
        
        Self { 
            api_key,  
            organization,
            base_path: API_BASE_PATH.to_string(),
            proxy: None,
            timeout: None,
        }
    }

    pub fn new_personal(api_key: String) -> Self {
        Self::new(Some(api_key), None)
    }

    pub fn new_organization(api_key: String, organization: String) -> Self {
        Self::new(Some(api_key), Some(organization))
    }

    pub fn base_path(mut self, base_path: String) -> Self {
        self.base_path = base_path;
        self
    }

    pub fn proxy(mut self, proxy: String) -> Self {
        self.proxy = Some(proxy);
        self
    }

    pub fn timeout(mut self, timeout: u64) -> Self {
        self.timeout = Some(timeout);
        self
    }

    pub fn apply_to_request(self, builder: reqwest::ClientBuilder, path: String, method: reqwest::Method) -> reqwest::RequestBuilder{
        let mut client_builder = match self.proxy.clone() {
            Some(proxy) => {
                let proxy = reqwest::Proxy::https(proxy).unwrap();
                builder.proxy(proxy)
            },
            None => builder,
        };
        client_builder = match self.timeout.clone() {
            Some(timeout) => client_builder.timeout(Duration::from_secs(timeout)),
            None => client_builder,
        };

        let client = client_builder.user_agent(USER_AGENT_VALUE).build().unwrap();    
        
        let url = format!("{}{}", self.base_path, path);
        let mut builder = client.request(method, url);
        builder = match self.api_key.clone() {
            Some(key) => builder.bearer_auth(key.clone()),
            None => builder,
        };
        // builder = builder.header("Content-Type", content_type);
        match self.organization.clone() {
            Some(org) => builder.header("OpenAI-Organization", org),
            None => builder,
        }
    }

}