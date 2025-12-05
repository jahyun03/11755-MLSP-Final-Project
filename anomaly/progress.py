from time import sleep
from rich.table import Column
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    ProgressColumn,
)
from rich.text import Text
import atexit
from typing import Optional


class RichColumn(ProgressColumn):
    def __init__(self, table_column: Optional[Column] = None) -> None:
        super().__init__(table_column)
        self.time_elapsed_column = TimeElapsedColumn()
        self.time_remaining_column = TimeRemainingColumn()
        self.m_of_n = MofNCompleteColumn()
        self._completed = 0
        self.sec_per_iter = -1
        self.info = None

    def render(self, task: "Task") -> Text:
        m_of_n = self.m_of_n.render(task)
        m_of_n = Text(f"{m_of_n}".replace(" ", ""), style="red")
        elapsed = self.time_elapsed_column.render(task)
        elapsed = Text(f"{elapsed}", style="orange_red1") + Text(
            "/", style="dark_orange"
        )
        remaining = self.time_remaining_column.render(task)
        remaining = Text(f"{remaining}", style="yellow")
        if task.completed:
            if self._completed < task.completed:
                # do not update sec_per_iter if no new completed iterators
                self._completed = task.completed
                self.sec_per_iter = task.elapsed / task.completed
            sec_per_iter = Text(f"({self.sec_per_iter:.1f}s/iter)", style="green")
        else:
            sec_per_iter = Text("(--s/iter)", style="green")

        rendered = m_of_n + " " + elapsed + remaining + sec_per_iter
        if self.info is None:
            return rendered
        info = Text(f" {self.info}", style="cyan")
        return rendered + info


info_column = RichColumn()
progress = Progress(
    TextColumn("[bold]{task.description}", table_column=Column(ratio=1)),
    BarColumn(bar_width=None, table_column=Column(ratio=8), complete_style="blue"),
    info_column,
    expand=True,
    redirect_stdout=False,
    redirect_stderr=False,
)


def track(*args, **kwargs):
    return progress.track(*args, **kwargs)


def exit_progress():
    progress.__exit__(None, None, None)


def start():
    progress.__enter__()
    atexit.register(exit_progress)


def stop():
    progress.__exit__(None, None, None)


if __name__ == "__main__":
    start()
    for n in track(range(10), description="Working...  "):
        sleep(0.01)
        print(n)
        if n == 8:
            0 / 0
    stop()
