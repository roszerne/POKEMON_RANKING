import itertools
import tkinter as tk
from tkinter import ttk
import requests, io
from PIL import ImageTk, Image
import data
import pandas as pd
import tkinter.messagebox


class Gui():
    def __init__(self, parent):
        self.parent = parent
        self.parent.geometry("600x600")

        self.url_images = 'https://img.pokemondb.net/artwork/'
        self.dataframe = None

        self.container = tk.Frame(parent)
        self.canvas = tk.Canvas(self.container, width=600, height=600)

        self.add_scrollbar()
        self.set_data(20)

        button1 = tk.Button(self.parent, text="OK", command=lambda: self.get_pokemons())
        button1.pack()

        # first window
        self.chosen_pokemons = []
        self.checkbuttons_var = {}
        self.checkbuttons_images = {}
        self.checkbuttons = {}

        # second window
        self.scale_buttons = {}
        self.scale_comboboxes = {}
        self.scale_var = {}
        self.chosen_scale = []
        self.stats = ['Attack', 'Defense', 'HP', 'Sp. Atk']
        self.scale = ['Equal importance', 'Somewhat more important', 'Much more important']

        self.set_checkboxes()
        self.container.pack()
        self.canvas.pack(side="left", fill="both", expand=True)

    def set_data(self, how_many):
        data.data()
        df = pd.read_json('PokemonData.json')
        df = df.set_index(['#'])
        self.dataframe = df.head(how_many)

    def add_scrollbar(self):
        scrollbar = tk.Scrollbar(self.container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

    def set_checkboxes(self):
        for i, name in enumerate(self.dataframe['Name']):
            # bo wtedy nie wiem jak strona wyglada
            if " " in name:
                continue
            image_url = 'https://img.pokemondb.net/artwork/' + name.lower() + '.jpg'
            response = requests.get(image_url)
            image = io.BytesIO(response.content)
            image = Image.open(image)
            image = image.resize((100, 100))
            pokemon_image = ImageTk.PhotoImage(image)

            var = tk.IntVar(0)
            self.checkbuttons_var[name] = var
            self.checkbuttons_images[name] = pokemon_image
            cb = tk.Checkbutton(self.scrollable_frame,
                                variable=var,
                                # command=lambda: self.cb_command(),
                                text="          " + name + "   ",
                                offvalue=0,
                                image=pokemon_image,
                                compound='left',
                                )
            cb.config(font=("Arial", 10))
            cb.grid(sticky="w")  # to keep it aligned
            self.checkbuttons[name] = cb

    # def cb_command(self):
    #     if not self.checkbuttons_var[name].get():
    #         self.checkbuttons_var[name].set(1)
    #     else:
    #         self.checkbuttons_var[name].set(0)

    def get_pokemons(self):
        for name in self.dataframe['Name']:
            if " " in name:
                continue
            if self.checkbuttons_var[name].get() and name not in self.chosen_pokemons:
                self.chosen_pokemons.append(name)
                print("Dodaj tego pokemona" + name)

        if len(self.chosen_pokemons) < 4 or len(self.chosen_pokemons) > 5:
            tkinter.messagebox.showinfo("Error", "Zła liczba")
            # self.chosen_pokemons = []
            # for k in self.checkbuttons_var:
            #     self.checkbuttons_var[k].set(0)
        else:
            self.open_scale_window()

    def open_scale_window(self):
        self.delete_old_window()
        self.parent.geometry("600x600")
        button1 = tk.Button(self.parent, text="OK", command=lambda: self.get_scale())
        button1.pack()

        i = 1
        for stats_pair in itertools.combinations(self.stats, 2):
            # print(stats_pair)
            self.create_buttons(stats_pair, i)
            self.create_combobox(stats_pair, i)
            i += 1

    def create_combobox(self, stat, i):
        self.parent.update()

        selected = tk.StringVar()
        wy = self.scale_buttons[stat][0].winfo_rooty()
        print(wy)
        scale_combobox = ttk.Combobox(self.parent, textvariable=selected, height=2, font=("Arial", 10), width=20)
        scale_combobox['values'] = self.scale
        scale_combobox['state'] = 'readonly'
        scale_combobox.place(x=400, y=wy / 2)
        self.scale_var[stat] = selected
        self.scale_comboboxes[stat] = scale_combobox

    def create_buttons(self, stat, i):
        # mozna dawac jakis obraz dac za przycisk i bedzie ladniej, jakis prostokat lepszy w sensie
        button1 = tk.Button(text=stat[0], height=2, width=10, font=("Arial", 10))
        button2 = tk.Button(text=stat[1], height=2, width=10, font=("Arial", 10))
        button1.place(x=20, y=80 * i)
        button2.place(x=200, y=80 * i)
        self.scale_buttons[stat] = (button1, button2)
        button1.configure(command=lambda: self.color_change(stat, 0))
        button2.configure(command=lambda: self.color_change(stat, 1))

    def color_change(self, stat, i):
        self.scale_buttons[stat][i].configure(bg='red')

    # tworze tablice ktora przyda sie potem - (wazniejsze kryterium, mniej wazne, i to co wybrane ze skali)
    def get_scale(self):
        for stats_pair in itertools.combinations(self.stats, 2):
            chosen = self.scale_var[stats_pair].get()
            if self.scale_buttons[stats_pair][0].cget('bg') == 'red':
                self.chosen_scale.append((stats_pair[0], stats_pair[1], chosen))
            else:
                self.chosen_scale.append((stats_pair[1], stats_pair[0], chosen))
        print(self.chosen_scale)

    def delete_old_window(self):
        self.canvas.delete('all')
        for widget in self.parent.winfo_children():
            widget.destroy()


root = tk.Tk()

gui = Gui(root)
root.mainloop()
