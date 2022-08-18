import tkinter as tk
from PIL import Image, ImageTk


class GUI(tk.Tk):
    def __init__(self, path_to_img):
        tk.Tk.__init__(self)
        self.grid()
        self.title("")
        self.main_grid = tk.Frame(
            self,
            bg="Red",
            bd=3,
            border=36,
            relief="raised",
            width=800,
            height=600
        )
        self.main_grid.pack(fill="both", expand=True)
        self.main_grid.grid_propagate(0)

        trash = []

        # get the photo
        img = ImageTk.PhotoImage(Image.open(path_to_img))
        label = tk.Label(self.main_grid, image=img)
        label.pack(anchor="n")
        trash.append([img, label])

        # text
        text_descr = "Predictions:"
        label = tk.Label(self.main_grid, text=text_descr, bg="red")
        label.pack(anchor="center")
        trash.append([label])

        # get the predictions
        text_prediction = "0%"
        label = tk.Label(self.main_grid, text=text_prediction, bg="red")
        label.pack(anchor="s")
        trash.append([label])
        self.mainloop()


GUI("../data_preparation/f2/nm0000203_23.png")
