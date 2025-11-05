use reqwest::Client;
use serde::{Deserialize, Serialize};
use yew::prelude::*;
use wasm_bindgen_futures::spawn_local;
use web_sys::{HtmlInputElement, Element};
use wasm_bindgen::JsCast;

// ===== DATA STRUCTURES =====

#[derive(Serialize, Debug, Clone)]
struct QueryRequest {
    query: String,
    top_k: u32,
    use_openai: bool,
}

#[derive(Deserialize, Debug, Clone)]
struct ApiResponse {
    #[allow(dead_code)]
    status: String,
    #[allow(dead_code)]
    query: String,
    answer: String,
    backend: Option<String>,
    #[allow(dead_code)]
    top_k: Option<u32>,
    #[allow(dead_code)]
    timestamp: Option<String>,
}

#[derive(Deserialize, Debug, Clone)]
struct ErrorResponse {
    #[allow(dead_code)]
    status: String,
    error: String,
    details: Option<String>,
    #[allow(dead_code)]
    timestamp: Option<String>,
}

// ===== MAIN APP COMPONENT =====

#[function_component(App)]
fn app() -> Html {
    let query = use_state(|| String::new());
    let answer = use_state(|| String::from("Your answer will appear here."));
    let answer_ref = use_node_ref();
    let is_loading = use_state(|| false);
    let error_message = use_state(|| None::<String>);
    let show_about = use_state(|| false);
    let show_contacts = use_state(|| false);
    let use_openai = use_state(|| false);
    let top_k = use_state(|| 5u32);
    let last_backend = use_state(|| String::from("None"));

    // Toggle between OpenAI and Ollama
    let toggle_backend = {
        let use_openai = use_openai.clone();
        Callback::from(move |e: Event| {
            if let Some(input) = e.target_dyn_into::<HtmlInputElement>() {
                use_openai.set(input.checked());
            }
        })
    };

    // Handle query input changes
    let on_input = {
        let query = query.clone();
        let error_message = error_message.clone();
        Callback::from(move |e: InputEvent| {
            if let Some(input) = e.target_dyn_into::<HtmlInputElement>() {
                query.set(input.value());
                // Clear error when user starts typing
                if error_message.is_some() {
                    error_message.set(None);
                }
            }
        })
    };

    // Handle top_k slider changes
    let on_top_k_change = {
        let top_k = top_k.clone();
        Callback::from(move |e: Event| {
            if let Some(input) = e.target_dyn_into::<HtmlInputElement>() {
                if let Ok(value) = input.value().parse::<u32>() {
                    top_k.set(value);
                }
            }
        })
    };

    // Handle form submission
    let on_submit = {
        let query = query.clone();
        let use_openai = use_openai.clone();
        let top_k = top_k.clone();
        let answer = answer.clone();
        let is_loading = is_loading.clone();
        let error_message = error_message.clone();
        let last_backend = last_backend.clone();

        Callback::from(move |e: SubmitEvent| {
            e.prevent_default();
            
            let query_value = (*query).clone().trim().to_string();
            
            // Validation
            if query_value.is_empty() {
                error_message.set(Some("Please enter a query.".to_string()));
                return;
            }

            if query_value.len() > 500 {
                error_message.set(Some("Query is too long (max 500 characters).".to_string()));
                return;
            }

            // Clear previous error and set loading
            error_message.set(None);
            is_loading.set(true);
            answer.set("Generating answer...".to_string());

            let client = Client::new();
            let answer = answer.clone();
            let is_loading = is_loading.clone();
            let error_message = error_message.clone();
            let use_openai_value = *use_openai;
            let top_k_value = *top_k;
            let last_backend = last_backend.clone();

            spawn_local(async move {
                let payload = QueryRequest {
                    query: query_value.clone(),
                    top_k: top_k_value,
                    use_openai: use_openai_value,
                };

                let response = client
                    .post("http://127.0.0.1:8000/generate/")
                    .json(&payload)
                    .send()
                    .await;

                match response {
                    Ok(res) => {
                        let status = res.status();
                        
                        if status.is_success() {
                            // Try to parse successful response
                            match res.json::<ApiResponse>().await {
                                Ok(api_response) => {
                                    // Log the response for debugging
                                    web_sys::console::log_1(&format!("Received answer: {}", &api_response.answer).into());
                                    answer.set(api_response.answer);
                                    if let Some(backend) = api_response.backend {
                                        last_backend.set(backend);
                                    }
                                    error_message.set(None);
                                }
                                Err(e) => {
                                    web_sys::console::error_1(&format!("Parse error: {:?}", e).into());
                                    error_message.set(Some(format!("Failed to parse response: {}", e)));
                                    answer.set("Error parsing response.".to_string());
                                }
                            }
                        } else {
                            // Try to parse error response
                            match res.json::<ErrorResponse>().await {
                                Ok(error_resp) => {
                                    let error_msg = if let Some(details) = error_resp.details {
                                        format!("{}: {}", error_resp.error, details)
                                    } else {
                                        error_resp.error
                                    };
                                    error_message.set(Some(error_msg));
                                    answer.set("Failed to generate answer.".to_string());
                                }
                                Err(e) => {
                                    error_message.set(Some(format!("Server error ({}): {}", status, e)));
                                    answer.set("Server error.".to_string());
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error_message.set(Some(format!("Connection error: {}. Make sure the API server is running.", e)));
                        answer.set("Error connecting to the API.".to_string());
                    }
                }

                is_loading.set(false);
            });
        })
    };

    // Toggle about section
    let on_about_click = {
        let show_about = show_about.clone();
        let show_contacts = show_contacts.clone();
        Callback::from(move |_| {
            show_about.set(!(*show_about));
            if *show_about {
                show_contacts.set(false);
            }
        })
    };

    // Toggle contacts section
    let on_contacts_click = {
        let show_contacts = show_contacts.clone();
        let show_about = show_about.clone();
        Callback::from(move |_| {
            show_contacts.set(!(*show_contacts));
            if *show_contacts {
                show_about.set(false);
            }
        })
    };

    // Effect to render HTML content
    {
        let answer = answer.clone();
        let answer_ref = answer_ref.clone();
        use_effect_with(answer.clone(), move |answer_val| {
            if let Some(element) = answer_ref.cast::<Element>() {
                element.set_inner_html(&**answer_val);
            }
            || ()
        });
    }

    html! {
        <div class="app-container">
            <header class="header">
                <h1>{ "🔍 RAG Historical Events Explorer" }</h1>
                <p class="subtitle">{ "Ask questions about 20th century historical events" }</p>
            </header>

            <main class="main-content">
                <div class="container">
                    // Backend Toggle
                    <div class="backend-toggle">
                        <label class="switch">
                            <input
                                type="checkbox"
                                id="backend-toggle"
                                name="backend-toggle"
                                checked={*use_openai}
                                onchange={toggle_backend}
                            />
                            <span class="slider"></span>
                        </label>
                        <label for="backend-toggle" class="toggle-label">
                            { if *use_openai {
                                "🤖 OpenAI GPT-4 Backend"
                            } else {
                                "🦙 Local Ollama Backend (phi3:mini)"
                            }}
                        </label>
                    </div>

                    // Query Form
                    <form class="query-form" onsubmit={on_submit}>
                        <div class="form-group">
                            <label for="query" class="form-label">
                                { "Enter your query:" }
                            </label>
                            <input
                                type="text"
                                id="query"
                                name="query"
                                class="form-input"
                                placeholder="e.g., When did World War II end?"
                                value={(*query).clone()}
                                oninput={on_input}
                                disabled={*is_loading}
                            />
                            <div class="char-counter">
                                { format!("{}/500 characters", (*query).len()) }
                            </div>
                        </div>

                        <div class="form-group">
                            <label for="top_k" class="form-label">
                                { format!("Number of documents to retrieve: {}", *top_k) }
                            </label>
                            <input
                                type="range"
                                id="top_k"
                                name="top_k"
                                class="form-slider"
                                min="1"
                                max="20"
                                value={top_k.to_string()}
                                onchange={on_top_k_change}
                                disabled={*is_loading}
                            />
                            <div class="slider-labels">
                                <span>{ "1" }</span>
                                <span>{ "20" }</span>
                            </div>
                        </div>

                        <button
                            type="submit"
                            class={classes!("submit-button", is_loading.then(|| "loading"))}
                            disabled={*is_loading}
                        >
                            { if *is_loading {
                                "⏳ Generating..."
                            } else {
                                "🚀 Submit Query"
                            }}
                        </button>
                    </form>

                    // Error Message
                    { if let Some(error) = (*error_message).clone() {
                        html! {
                            <div class="error-message">
                                <span class="error-icon">{ "⚠️" }</span>
                                <span>{ error }</span>
                            </div>
                        }
                    } else {
                        html! {}
                    }}

                    // Answer Section
                    <div class="answer-section">
                        <div class="answer-header">
                            <h2>{ "Answer:" }</h2>
                            { if *last_backend != "None" {
                                html! {
                                    <span class="backend-badge">
                                        { format!("Generated by: {}", *last_backend) }
                                    </span>
                                }
                            } else {
                                html! {}
                            }}
                        </div>
                        
                        { if *is_loading {
                            html! {
                                <div class="loading-spinner">
                                    <div class="spinner"></div>
                                    <p>{ "Loading..." }</p>
                                </div>
                            }
                        } else {
                            html! {
                                <div class="answer-content" ref={answer_ref}>
                                    // Content will be set by use_effect
                                </div>
                            }
                        }}
                    </div>

                    // Action Buttons
                    <div class="action-buttons">
                        <button
                            class={classes!("action-button", show_about.then(|| "active"))}
                            onclick={on_about_click}
                        >
                            { "ℹ️ About" }
                        </button>
                        <button
                            class={classes!("action-button", show_contacts.then(|| "active"))}
                            onclick={on_contacts_click}
                        >
                            { "📧 Contacts" }
                        </button>
                    </div>

                    // About Section
                    { if *show_about {
                        html! {
                            <div class="info-section about">
                                <h2>{ "About This Project" }</h2>
                                <p>{ "This is a Retrieval-Augmented Generation (RAG) system built with modern technologies:" }</p>
                                <ul>
                                    <li><strong>{ "Frontend:" }</strong>{ " Rust + Yew (WebAssembly)" }</li>
                                    <li><strong>{ "Backend:" }</strong>{ " Python + Flask" }</li>
                                    <li><strong>{ "Data Source:" }</strong>{ " DBpedia (20th century historical events)" }</li>
                                    <li><strong>{ "Embeddings:" }</strong>{ " sentence-transformers (all-MiniLM-L6-v2)" }</li>
                                    <li><strong>{ "Vector Search:" }</strong>{ " FAISS" }</li>
                                    <li><strong>{ "LLMs:" }</strong>{ " Ollama (phi3:mini) or OpenAI (GPT-4)" }</li>
                                </ul>
                                <p><strong>{ "Version:" }</strong>{ " 1.0.0" }</p>
                                <p class="feature-note">
                                    { "The system retrieves relevant historical documents and uses them as context for generating accurate, grounded answers." }
                                </p>
                            </div>
                        }
                    } else {
                        html! {}
                    }}

                    // Contacts Section
                    { if *show_contacts {
                        html! {
                            <div class="info-section contacts">
                                <h2>{ "Contact the Developers" }</h2>
                                <div class="contact-grid">
                                    <div class="contact-card">
                                        <h3>{ "👨‍💻 Francesco" }</h3>
                                        <p>
                                            <a href="https://github.com/frontinus" target="_blank" rel="noopener noreferrer">
                                                { "🐙 GitHub" }
                                            </a>
                                        </p>
                                        <p>
                                            <a href="https://linkedin.com/in/francesco-abate-79601719b" target="_blank" rel="noopener noreferrer">
                                                { "💼 LinkedIn" }
                                            </a>
                                        </p>
                                    </div>
                                    <div class="contact-card">
                                        <h3>{ "👨‍💻 Thomas" }</h3>
                                        <p>
                                            <a href="https://github.com/thetom061" target="_blank" rel="noopener noreferrer">
                                                { "🐙 GitHub" }
                                            </a>
                                        </p>
                                        <p>
                                            <a href="https://www.linkedin.com/in/thomas-cotte-9870531a1/" target="_blank" rel="noopener noreferrer">
                                                { "💼 LinkedIn" }
                                            </a>
                                        </p>
                                    </div>
                                </div>
                            </div>
                        }
                    } else {
                        html! {}
                    }}
                </div>
            </main>

            <footer class="footer">
                <p>{ "Built with ❤️ using Rust, Python, and AI" }</p>
            </footer>
        </div>
    }
}

fn main() {
    yew::Renderer::<App>::new().render();
}