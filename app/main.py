import typer
from ollama import chat
from ollama import ChatResponse



app = typer.Typer()

@app.command()
def hello(name: str):
    print(f"Hello {name}")


@app.command()
def leetcoach(name: str, solve: bool = False):
    if solve:
        print(f"Welcome to mas leetcoach! Lets solve.")
    else:
        print(f"Hello, I hope you are enjoying leetcoach!")

@app.command()
def ask(prompt: str):
    response: ChatResponse = chat(
            model="llama3.1:latest",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    print(response.message.content)

if __name__ == "__main__":
    app()
