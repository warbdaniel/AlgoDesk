"""adam ask - Query Claude API agent."""

import click
from lib.api_client import ClaudeAPI
from lib.formatter import console, print_error


@click.command()
@click.argument("question", nargs=-1, required=True)
@click.option("--no-stream", is_flag=True, help="Disable streaming")
def ask(question, no_stream):
    """Ask Claude API agent a question about trading."""
    query = " ".join(question)
    api = ClaudeAPI()

    if no_stream:
        data, err = api.query(query)
        if err:
            print_error(f"Claude API: {err}")
            return
        response = data.get("response", data.get("message", data.get("answer", str(data))))
        console.print(response)
    else:
        resp, err = api.query_stream(query)
        if err:
            print_error(f"Claude API: {err}")
            return

        try:
            for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    console.print(chunk, end="")
            console.print()
        except Exception as e:
            print_error(f"Stream error: {e}")
        finally:
            resp.close()
