from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import grid, vuetify

server = get_server()
state = server.state

LAYOUT = [
    {"x": 0, "y": 0, "w": 2, "h": 2, "i": "0"},
    {"x": 2, "y": 0, "w": 2, "h": 4, "i": "1"},
]

with SinglePageLayout(server) as layout:
    layout.title.set_text("Grid layout")
    with layout.content:
        with grid.GridLayout(
            layout=("layout", LAYOUT),
        ):
            grid.GridItem(
                "{{ item.i }}",
                v_for="item in layout",
                key="item.i",
                v_bind="item",
                classes="pa-4",
                style="border: solid 1px #333; background: rgba(128, 128, 128, 0.5);",
            )

if __name__ == "__main__":
    server.start()
