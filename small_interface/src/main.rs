use reqwest::Client;
use serde::{Deserialize, Serialize};
use yew::prelude::*;
use wasm_bindgen_futures::spawn_local;
use web_sys::HtmlInputElement;
use std::rc::Rc;
use std::cell::RefCell;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct QueryRequest {
    query: String,
    top_k: u32,
    use_openai: bool,
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
    let show_about = use_state(|| false);
    let show_contacts = use_state(|| false);
    let use_openai = use_state(|| Rc::new(false));

    let toggle_backend = {
        let use_openai = use_openai.clone();
        Callback::from(move |e: Event| {
            if let Some(input) = e.target_dyn_into::<HtmlInputElement>() {
                let new_state  = Rc::new(input.checked());
                use_openai.set(new_state);
            }
        })
    };

    let on_input = {
        let query = query.clone();
        Callback::from(move |e: InputEvent| {
            if let Some(input) = e.target_dyn_into::<HtmlInputElement>() {
                query.set(input.value());
            }
        })
    };

    let on_submit = {
        let query = query.clone();
        let use_openai = use_openai.clone();
        let answer = Rc::new(RefCell::new(answer.clone()));
        let is_loading = Rc::new(RefCell::new(is_loading.clone()));
        

        

        Callback::from(move |e: SubmitEvent| {
            e.prevent_default(); 
            let query_value = (*query).clone();
            if query_value.is_empty() {
                answer.borrow().set("Please enter a query.".to_string());
                return;
            }

            is_loading.borrow().set(true);
            let client = Client::new();
            let answer = answer.clone();
            let is_loading = is_loading.clone();
            let use_openai = Rc::clone(&use_openai);

            spawn_local(async move {
                let payload = QueryRequest {
                    query: query_value.clone(),
                    top_k: 5,
                    use_openai:*use_openai,
                };

                let response = client
                    .post("http://127.0.0.1:8000/generate/")
                    .json(&payload)
                    .send()
                    .await;

                match response {
                    Ok(res) => {
                        if let Ok(api_response) = res.json::<ApiResponse>().await {
                            answer.borrow().set(api_response.answer);
                        } else {
                            answer.borrow().set("Failed to parse response.".to_string());
                        }
                    }
                    Err(_) => answer.borrow().set("Error connecting to the API.".to_string()),
                }

                is_loading.borrow().set(false);
            });
        })
    };

    let on_about_click = {
        let show_about = show_about.clone();
        Callback::from(move |_| {
            show_about.set(!(*show_about));
        })
    };

    let on_contacts_click = {
        let show_contacts = show_contacts.clone();
        Callback::from(move |_| {
            show_contacts.set(!(*show_contacts));
        })
    };

    html! {
        <div style="font-family: Arial, sans-serif; padding: 2rem;">
            <h1>{ "RAG Front-End in Rust" }</h1>
            <div>
                <input
                    type="checkbox"
                    id="backend-toggle"
                    name="backend-toggle"
                    onchange={toggle_backend}
                />
                <label for="backend-toggle" style="margin-left: 0.5rem;">
                    { if **use_openai { "OpenAI backend selected." } else { "Local AI backend selected." } }
                </label>
            </div>
            <form onsubmit={on_submit}>
                <label for="query">{ "Enter your query:" }</label>
                <input
                    type="text"
                    id="query"
                    name="query"
                    value={(*query).clone()}
                    oninput={on_input}
                />
                <button type="submit">
                    { "Submit" }
                </button>
            </form>
            <div class="answer">
                <strong>{ "Answer:" }</strong>
                { if *is_loading {
                    html! { <p>{ "Loading..." }</p> }
                } else {
                    html! { <div style="white-space: pre-wrap;">{ (*answer).clone() }</div> }
                }}
            </div>
            <button onclick={on_about_click} >
                { "About" }
            </button>
            { if *show_about {
                html! {
                    <div class="about">
                        <h2>{ "About" }</h2>
                        <p>{ "This is a simple front-end for the RAG API written in Rust using Yew." }</p>
                        <p>{ "The RAG API is a Retrieval-Augmented Generation model that can be used to generate answers to queries." }</p>
                        <p>{ "The model is fine-tuned on the Wikipedia dataset and can generate answers based on the input query." }</p>
                        <p>{ "The front-end sends the query to the API and displays the generated answer." }</p>
                        <p>{ "version: 1.0.0" }</p>
                    </div>
                }
            } else {
                html! {}
            }}
            <button onclick={on_contacts_click} >
                { "Contacts" }
            </button>
            { if *show_contacts {
                html! {
                    <div class="contacts">
                        <h2>{ "Contacts" }</h2>
                        <p>{ "Keep in touch with us:" }</p>
                        <p>{ "Francesco's GitHub: " }<a href="https://github.com/frontinus" target="_blank">{ "Francesco's GitHub" }</a></p>
                        <p>{ "Thomas' GitHub: " }<a href="https://github.com/thetom061" target="_blank">{ "Thomas' GitHub" }</a></p>
                        <p>{ "Francesco's LinkedIn: " }<a href="https://linkedin.com/in/francesco-abate-79601719b" target="_blank">{ "Francesco's LinkedIn" }</a></p>
                        <p>{ "Thomas' LinkedIn: " }<a href="https://www.linkedin.com/in/thomas-cotte-9870531a1/" target="_blank">{ "Thomas' LinkedIn" }</a></p>
                        </div>
                }
            } else {
                html! {}
            }}
        </div>
    }
}

fn main() {
    yew::Renderer::<App>::new().render();
}