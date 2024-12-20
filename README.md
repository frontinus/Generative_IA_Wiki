
Setup Script - installs python requirements + gathers data for the project


Model uses Ollama on a small wikipedia dump from the 20th century.


It is built with a rust interface and flask backend with python

How to use:

1)run ollama fron cli

2)run app.py from cli

3) run rust interface by doing
3) a) cd small_interface
3) b) trunk serve
3) c) open http://127.0.0.1:8080/ in your navigator

4) write a prompt and send it, then wait for an answer
