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

            spawn_local(async move {
                let payload = QueryRequest {
                    query: query_value.clone(),
                    top_k: 5,
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
                    html! { <div style="white-space: pre-wrap;">{ (*answer).clone() }</div> }
                }}
            </div>
            <button onclick={on_about_click} style="margin-top: 2rem; padding: 0.5rem;">
                { "About" }
            </button>
            { if *show_about {
                html! {
                    <div style="margin-top: 1rem; font-size: 1.2rem;">
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
            <button onclick={on_contacts_click} style="margin-top: 2rem; padding: 0.5rem;">
                { "Contacts" }
            </button>
            { if *show_contacts {
                html! {
                    <div style="margin-top: 1rem; font-size: 1.2rem;">
                        <h2>{ "Contacts" }</h2>
                        <p>{ "Keep in touch with us:" }</p>
                        <p>{ "Francesco's GitHub: " }<a href="https://github.com/frontinus" target="_blank">{ "Francesco's GitHub" }</a></p>
                        <p>{ "Thomas' GitHub: " }<a href="https://github.com/thetom061" target="_blank">{ "Thomas' GitHub" }</a></p>
                        <p>{ "Francesco's LinkedIn: " }<a href="https://linkedin.com/in/francesco-abate-79601719b" target="_blank">{ "Francesco's LinkedIn" }</a></p>
                        <p>{ "Thomas' LinkedIn: " }<a href="https://www.linkedin.com/in/thomas-cotte-9870531a1/" target="_blank">{ "Thomas' LinkedIn" }</a></p><p>{ "Francesco Abate" }</p>
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