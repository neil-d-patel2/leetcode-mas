import typer

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


if __name__ == "__main__":
    app()
