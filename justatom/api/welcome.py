import inspect
import random
from functools import partial, wraps

import anyio
import asyncer
import pyfiglet as pfg
from loguru import logger
from typer import Typer

# Typer doesn't support the `in-place` asyncio support. Hence we wrap around it to make it coro-friendly.
# See https://github.com/tiangolo/typer/issues/88#issuecomment-1732469681


class AsyncTyper(Typer):
    @staticmethod
    def maybe_run_async(decorator, f):
        if inspect.iscoroutinefunction(f):

            @wraps(f)
            def runner(*args, **kwargs):
                return asyncer.runnify(f)(*args, **kwargs)

            decorator(runner)
        else:
            decorator(f)
        return f

    def callback(self, *args, **kwargs):
        decorator = super().callback(*args, **kwargs)
        return partial(self.maybe_run_async, decorator)

    def command(self, *args, **kwargs):
        decorator = super().command(*args, **kwargs)
        return partial(self.maybe_run_async, decorator)


cli = AsyncTyper()


@cli.command()
async def main(fonts: list[str] = None, text: str = "Welcome"):
    fonts = random.sample(pfg.FigletFont.getFonts(), k=10)
    # sub-zero is the most visually attractive!
    logger.info("\n".join(fonts))
    for font in fonts:
        _font = str(font)
        logger.info(f"FONT {_font}\n")
        f = pfg.Figlet(font=_font)
        message = f.renderText(text)
        logger.info(message + "\n\n\n")
        await anyio.sleep(0.1)

    _font = "sub-zero"
    logger.info(f"FONT {_font}\n")
    f = pfg.Figlet(font=_font, width=128)
    message = f.renderText("Welcome")
    logger.info(message + "\n\n\n")
    await anyio.wait_all_tasks_blocked()


if __name__ == "__main__":
    cli()
