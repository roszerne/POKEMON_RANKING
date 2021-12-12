import tkinter as tk
from tkinter import ttk

import io
import itertools
import numpy as np
import pandas as pd
import requests
import tkinter.messagebox
from PIL import ImageTk, Image

import data
from Pokemon import Pokemon
from ranking import Ranking


class Gui:
    def __init__(self, parent):
        self.size = 600
        self.parent = parent
        self.parent.geometry("{}x{}".format(self.size, self.size))
        self.parent.config(bg='white')
        self.parent.resizable(False, False)

        self.url_images = 'https://img.pokemondb.net/artwork/'
        self.dataframe = None

        self.container = tk.Frame(parent, bg='white')
        self.canvas = tk.Canvas(self.container, width=self.size, height=self.size, bg='white', highlightthickness=0)

        self.set_data(10)

        self.scrollable_frame = None
        self.add_scrollbar()
        self.chosen_pokemons = []
        self.checkbuttons_var = {}
        self.checkbuttons_images = {}
        self.checkbuttons = {}

        # second window
        self.scale_buttons = {}
        self.scale_comboboxes = {}
        self.scale_var = {}
        self.scale_color = '#309BDC'
        self.stats = ['HP', 'Attack', 'Defense', 'Speed']
        self.scale = ['Equal importance', 'Somewhat more important', 'Much more important', 'Very much important',
                      'Absolutely more important']
        self.chosen_scale = np.ones((len(self.stats), len(self.stats)), dtype='double')  # macierz kryteriów lvl 2
        self.ranking = []

        self.incomplete_data = False
        self.method = ''

        self.set_checkboxes()
        self.set_ok_button(1)
        self.container.pack()
        self.canvas.pack(side="left", fill="both", expand=True)

    def set_data(self, how_many):
        data.data(how_many)
        df = pd.read_json('PokemonData.json')
        df = df.set_index(['#'])
        self.dataframe = df.head(how_many)
        print(self.dataframe)

    def set_ok_button(self, i):
        ok_button = tk.PhotoImage(file="okbutton--1-.png")
        img_label = tk.Label(image=ok_button)
        img_label.image = ok_button
        if i == 1:
            button1 = tk.Button(self.parent, image=ok_button, command=lambda: self.get_pokemons(), borderwidth=0,
                                bg='white')
        elif i ==2:
            button1 = tk.Button(self.parent, image=ok_button, command=lambda: self.get_scale(), borderwidth=0,
                                bg='white')
        else:
            button1 = tk.Button(self.parent, image=ok_button, command=lambda: self.start_ranking(), borderwidth=0,
                                bg='white')

        button1.pack()

    def add_scrollbar(self):

        scrollbar = tk.Scrollbar(self.container, orient="vertical", command=self.canvas.yview)

        self.scrollable_frame = tk.Frame(self.canvas, bg='white')
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")

    def set_checkboxes(self):
        for i, name in enumerate(self.dataframe['Name']):
            if " " in name:
                continue
            if "\'" in name:
                name = name.replace("\'", "")
            image_url = 'https://img.pokemondb.net/artwork/' + name.lower() + '.jpg'
            response = requests.get(image_url)
            image = io.BytesIO(response.content)
            image = Image.open(image).resize((110, 110))
            pokemon_image = ImageTk.PhotoImage(image)

            var = tk.IntVar(0)
            self.checkbuttons_var[name] = var
            self.checkbuttons_images[name] = pokemon_image
            cb = tk.Checkbutton(self.scrollable_frame,
                                variable=var,
                                text=self.get_checkbutton_text(i),
                                offvalue=0,
                                image=pokemon_image,
                                compound='left',
                                bg='white',
                                font=("Arial", 11)
                                )
            cb.grid(sticky="w")  # to keep it aligned
            self.checkbuttons[name] = cb

    def get_pokemons(self):
        for index, row in self.dataframe.iterrows():
            name = row['Name']
            if " " in name:
                continue
            if self.checkbuttons_var[name].get() and not any(pokemon.name == name for pokemon in self.chosen_pokemons):
                self.chosen_pokemons.append(Pokemon(index, row))  # tworze obiekt klasy Pokemon

        # if len(self.chosen_pokemons) < 5 or len(self.chosen_pokemons) > 9:
        #     tkinter.messagebox.showinfo("Error", "Wrong number")
        # else:
        #     self.open_scale_window()

        self.open_scale_window() # nie ma ograniczenia liczbowego juz

    def get_checkbutton_text(self, i):
        res = f"{str(self.dataframe.iloc[i][0]).replace(' ', '') : ^110}"
        res += "\n\n"
        for j in range(4):
            res += f"{self.stats[j]}: {str(self.dataframe.iloc[i][j + 1]) : ^5}" + "  "
        return res

    def open_scale_window(self):

        self.delete_old_window()
        self.parent.geometry("{}x{}".format(self.size, self.size))
        self.set_ok_button(2)

        i = 1
        for stats_pair in itertools.combinations(self.stats, 2):
            self.create_buttons(stats_pair, i)
            self.create_combobox(stats_pair, i)
            i += 1

    def create_combobox(self, stat, i):
        self.parent.update()

        selected = tk.StringVar()
        scale_combobox = ttk.Combobox(self.parent, textvariable=selected, height=4, font=("Arial", 10), width=20)
        scale_combobox['values'] = self.scale
        scale_combobox['state'] = 'readonly'
        y_combobox = 10 + 83.5 * i
        scale_combobox.place(x=400, y=y_combobox)
        self.scale_var[stat] = selected
        self.scale_comboboxes[stat] = scale_combobox

    # przyciski z kryteriami
    def create_buttons(self, stat, i):
        button1 = tk.Button(text=stat[0], height=1, width=15, font=("Arial", 10), bg='white', bd=3, relief='ridge')
        button2 = tk.Button(text=stat[1], height=1, width=15, font=("Arial", 10), bg='white', bd=3, relief='ridge')
        button1.place(x=20, y=85 * i)
        button2.place(x=200, y=85 * i)
        self.scale_buttons[stat] = (button1, button2)
        button1.configure(command=lambda: self.color_change(stat, 0))
        button2.configure(command=lambda: self.color_change(stat, 1))

    def color_change(self, stat, i):
        self.scale_buttons[stat][i].configure(bg=self.scale_color)
        self.scale_buttons[stat][(i + 1) % 2].configure(bg='white')

    def get_scale(self):
        for stats_pair in itertools.combinations(self.stats, 2):

            chosen = self.scale_var[stats_pair].get()
            if not chosen:  # nie zaznaczone nic w comboboxie
                print("None!!!!Combobox ", stats_pair)
                self.incomplete_data = True
                ind1 = self.stats.index(stats_pair[1])
                ind2 = self.stats.index(stats_pair[0])
                self.chosen_scale[ind1, ind2] = None
                self.chosen_scale[ind2, ind1] = None
                # tkinter.messagebox.showinfo("Error", "Choose all")
                break

            elif self.scale_buttons[stats_pair][0].cget('bg') == self.scale_color:
                val = self.scale.index(chosen)
                val = val * 2 + 1
                ind1 = self.stats.index(stats_pair[0])
                ind2 = self.stats.index(stats_pair[1])
                self.chosen_scale[ind1, ind2] = val
                self.chosen_scale[ind2, ind1] = 1 / val

            elif self.scale_buttons[stats_pair][1].cget('bg') == self.scale_color:
                val = self.scale.index(chosen)
                val = val * 2 + 1
                ind1 = self.stats.index(stats_pair[1])
                ind2 = self.stats.index(stats_pair[0])
                self.chosen_scale[ind1, ind2] = val
                self.chosen_scale[ind2, ind1] = 1 / val

            else:
                self.incomplete_data = True # tutaj tego przycisku nie zaznaczylismy
                print("None!!!!BUTTON ", stats_pair)
                ind1 = self.stats.index(stats_pair[1])
                ind2 = self.stats.index(stats_pair[0])
                self.chosen_scale[ind1, ind2] = None
                self.chosen_scale[ind2, ind1] = None
                # tkinter.messagebox.showinfo("Error", "Choose all")
                break

        self.open_options_window()

    def open_options_window(self):
        for widget in self.parent.winfo_children():
            widget.destroy()

        self.set_ok_button(3)

        self.varAHP = tk.IntVar(0)
        self.varGMM = tk.IntVar(0)
        cb = tk.Checkbutton(self.parent,
                            variable=self.varAHP,
                            text="AHP",
                            offvalue=0,
                            compound='left',
                            bg='white',
                            font=("Arial", 11)
                            )
        cb.pack()

        cb = tk.Checkbutton(self.parent,
                            variable=self.varGMM,
                            text="GMM",
                            offvalue=0,
                            compound='left',
                            bg='white',
                            font=("Arial", 11)
                            )
        cb.pack()

    def start_ranking(self):
        if self.varAHP.get():
            self.method = 'AHP'
        elif self.varGMM.get():
            self.method = 'GMM'

        rank = Ranking(self.chosen_pokemons, self.chosen_scale, self.method, self.incomplete_data)
        self.ranking = rank.AHP()

        self.open_ranking_window()

    def open_ranking_window(self):
        for widget in self.parent.winfo_children():
            widget.destroy()

        self.container = tk.Frame(self.parent, bg='white')
        self.canvas = tk.Canvas(self.container, width=self.size, height=self.size, bg='white', highlightthickness=0)
        self.add_scrollbar()

        numbers = []
        sorted_ranking = sorted(self.ranking, reverse=True)
        pokemon_images = []

        for i in range(len(self.chosen_pokemons)):
            result = sorted_ranking[i]
            idx, = np.where(self.ranking == result)
            pokemon = self.chosen_pokemons[idx[0]]
            font_size = 13

            numbers.append(
                tk.Label(self.scrollable_frame, text="{}".format(i + 1), font=("Arial", font_size), bg='white', bd=0))
            numbers[i].grid(column=1, row=i + 1, padx=10)

            pokemon_images_label = tk.Label(self.scrollable_frame, image=self.checkbuttons_images[pokemon.name], bd=0)
            pokemon_images.append(pokemon_images_label)
            pokemon_images[i].grid(column=2, row=i + 1, padx=10, pady=10)

            ranking_button = tk.Button(self.scrollable_frame, text=pokemon.name + ':   ' + str(round(result, 3)),
                                       font=("Arial", font_size),
                                       bg='white', bd=0)
            ranking_button.grid(column=3, row=i + 1, padx=10)

            stats_label = tk.Button(self.scrollable_frame, text=pokemon.df[1:].to_string(),
                                    font=("Arial", font_size - 2),
                                    bg='white', bd=0)
            stats_label.grid(column=5, row=i + 1, padx=50)

        self.container.pack()
        self.canvas.pack(side="left", fill="both", expand=True)

    def delete_old_window(self):
        self.canvas.delete('all')
        for widget in self.parent.winfo_children():
            widget.destroy()


root = tk.Tk()
root.title("Pokemon Ranking")
gui = Gui(root)
root.mainloop()
