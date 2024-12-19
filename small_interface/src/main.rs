use reqwest::Client;
use serde::{Deserialize, Serialize};
use yew::prelude::*;
use wasm_bindgen_futures::spawn_local;
use web_sys::HtmlInputElement;


#[derive(Serialize, Deserialize, Debug, Clone)]
struct QueryRequest {
    query: String,
    top_k: u32,
}

#[derive(Deserialize, Debug, Clone)]
struct ApiResponse {
    query: String,
    answer: String,
}

#[function_component(App)]
fn app() -> Html {
    let query = use_state(|| String::new());
    let answer = use_state(|| String::from("Your answer will appear here."));
    let is_loading = use_state(|| false);

    let on_input = {
        let query = query.clone();
        Callback::from(move |e: InputEvent| {
            if let Some(input) = e.target_dyn_into::<HtmlInputElement>() {
                query.set(input.value());
            }
        })
    };

    // Updated to use SubmitEvent instead of FocusEvent
    let on_submit = {
        let query = query.clone();
        let answer = answer.clone();
        let is_loading = is_loading.clone();

        Callback::from(move |e: SubmitEvent| {
            e.prevent_default();  // Prevent the default form submission behavior
            let query_value = (*query).clone();
            if query_value.is_empty() {
                answer.set("Please enter a query.".to_string());
                return;
            }

            is_loading.set(true);
            let client = Client::new();
            spawn_local(async move {
                let payload = QueryRequest {
                    query: query_value.clone(),
                    top_k: 5,
                };

                let response = client
                    .post("http://127.0.0.1:8000/generate/")
                    .json(&payload)  // Ensure that .json() works here
                    .send()
                    .await;

                match response {
                    Ok(res) => {
                        if let Ok(api_response) = res.json::<ApiResponse>().await {
                            answer.set(api_response.answer);
                        } else {
                            answer.set("Failed to parse response.".to_string());
                        }
                    }
                    Err(_) => answer.set("Error connecting to the API.".to_string()),
                }

                is_loading.set(false);
            });
        })
    };

    html! {
        <div style="font-family: Arial, sans-serif; padding: 2rem;">
            <h1>{ "RAG Front-End in Rust" }</h1>
            <form onsubmit={on_submit}>
                <label for="query">{ "Enter your query:" }</label>
                <input
                    type="text"
                    id="query"
                    name="query"
                    value={(*query).clone()}
                    oninput={on_input}
                    style="margin-left: 1rem; padding: 0.5rem;"
                />
                <button type="submit" style="margin-left: 1rem; padding: 0.5rem;">
                    { "Submit" }
                </button>
            </form>
            <div style="margin-top: 2rem; font-size: 1.2rem;">
                <strong>{ "Answer:" }</strong>
                { if *is_loading {
                    html! { <p>{ "Loading..." }</p> }
                } else {
                    html! { <p>{ (*answer).clone() }</p> }
                }}
            </div>
        </div>
    }
}


fn main() {
    yew::Renderer::<App>::new().render();
}
