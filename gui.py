import tkinter as tk
from tkinter import ttk

import io
import itertools
import numpy as np
import pandas as pd
import requests
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
        self.checkboxes_var = {}
        self.checkboxes_images = {}
        self.checkboxes = {}

        # second window
        self.scale_buttons = {}
        self.scale_comboboxes = {}
        self.scale_var = {}
        self.scale_color = '#309BDC'
        self.stats = ['HP', 'Attack', 'Defense', 'Sp Attack', 'Sp Defense', 'Speed']  # stats in data
        self.scale = ['Equal importance', 'Somewhat more important', 'Much more important', 'Very much important',
                      'Absolutely more important']  # AHP scale
        self.subcriteria = [('Sp Attack', 'Sp Defense'),
                            ("HP", "Defense")]  # subcriteria 1st tuple = special, 2nd tuple = endurance
        self.criteria = ['Endurance', 'Special', 'Attack', 'Speed']  # main criteria
        self.chosen_scale = None
        self.subscale = None

        self.ranking = []
        self.entry = None
        self.experts = 1 # number of experts, default is 1
        self.varEVM = tk.IntVar(0)
        self.varGMM = tk.IntVar(0)
        self.incomplete_data = False
        self.method = 'EVM'  # default method

        # create 1st window
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

    def set_ok_button(self, i, exp_num=0):
        ok_button = tk.PhotoImage(file="okbutton--1-.png")
        img_label = tk.Label(image=ok_button)
        img_label.image = ok_button
        if i == 1:  # after 1st window
            button1 = tk.Button(self.parent, image=ok_button, command=lambda: self.get_pokemons(), borderwidth=0,
                                bg='white')
        elif i == 2: # after window with buttons and checkboxes, exp_num is the number of choosing expert
            button1 = tk.Button(self.parent, image=ok_button, command=lambda: self.get_scale(exp_num), borderwidth=0,
                                bg='white')
        else: # after the option window, getting the number of experts and which method is to be used
            button1 = tk.Button(self.parent, image=ok_button, command=lambda: self.get_experts(), borderwidth=0,
                                bg='white')
        button1.pack()

    def add_scrollbar(self):

        scrollbar = tk.Scrollbar(self.container, orient="vertical", command=self.canvas.yview)

        self.scrollable_frame = tk.Frame(self.canvas, bg='white')
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")

    # checkboxes in 1st window
    def set_checkboxes(self):
        for i, name in enumerate(self.dataframe['Name']):
            if " " in name:
                continue
            if "\'" in name:
                name = name.replace("\'", "")
            image_url = 'https://img.pokemondb.net/artwork/' + name.lower() + '.jpg'  # getting the image
            response = requests.get(image_url)
            image = io.BytesIO(response.content)
            image = Image.open(image).resize((110, 110))
            pokemon_image = ImageTk.PhotoImage(image)

            var = tk.IntVar(0)
            self.checkboxes_var[name] = var
            self.checkboxes_images[name] = pokemon_image
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
            self.checkboxes[name] = cb

    def get_pokemons(self):
        for index, row in self.dataframe.iterrows():
            name = row['Name']
            if " " in name:
                continue
            if self.checkboxes_var[name].get() and not any(pokemon.name == name for pokemon in self.chosen_pokemons):
                self.chosen_pokemons.append(Pokemon(index, row))  # Pokemon class
        self.open_options_window()

    def get_checkbutton_text(self, i):
        res = f"{str(self.dataframe.iloc[i][0]).replace(' ', '') : ^110}"
        res += "\n\n"
        for j in range(len(self.stats)):
            res += f"{self.stats[j]}: {str(self.dataframe.iloc[i][j + 1]) : ^5}" + "  "
            if j == 2:
                res += "\n"
        return res

    def get_experts(self):
        self.experts = int(self.entry.get())
        # creating matrices after getting the num od experts
        self.chosen_scale = np.ones((self.experts, len(self.criteria), len(self.criteria)),
                                    dtype='double')
        self.subscale = np.ones((self.experts, 2, 2, 2), dtype='double')

        # scale window for first expert
        self.open_scale_window(0)

    def open_scale_window(self, expert_number):
        for widget in self.parent.winfo_children():
            widget.destroy()

        self.container = tk.Frame(self.parent, bg='white')
        self.canvas = tk.Canvas(self.container, width=self.size, height=self.size, bg='white', highlightthickness=0)
        self.add_scrollbar()
        self.set_ok_button(2, expert_number)

        i = 1
        for stats_pair in itertools.combinations(self.criteria, 2):
            self.create_buttons(stats_pair, i)
            self.create_combobox(stats_pair, i)
            i += 1
        for sub in self.subcriteria:
            self.create_buttons(sub, i)
            self.create_combobox(sub, i)
            i += 1

        self.container.pack()
        self.canvas.pack()

    def create_combobox(self, stat, i):
        self.scrollable_frame.update()

        selected = tk.StringVar()
        scale_combobox = ttk.Combobox(self.scrollable_frame, textvariable=selected, height=4, font=("Arial", 11),
                                      width=20)
        scale_combobox['values'] = self.scale
        scale_combobox.current(0)
        scale_combobox['state'] = 'readonly'
        scale_combobox.grid(column=8, row=i + 2, padx=20, pady=20)
        self.scale_var[stat] = selected
        self.scale_comboboxes[stat] = scale_combobox

    # buttons for crietria and subcriteria
    def create_buttons(self, stat, i):
        button1 = tk.Button(self.scrollable_frame, text=stat[0], height=1, width=15, font=("Arial", 11), bg='white',
                            bd=3, relief='ridge')
        button2 = tk.Button(self.scrollable_frame, text=stat[1], height=1, width=15, font=("Arial", 11), bg='white',
                            bd=3, relief='ridge')
        button1.grid(column=3, row=i + 2, padx=20, pady=20)
        button2.grid(column=5, row=i + 2, padx=20, pady=20)
        self.scale_buttons[stat] = (button1, button2)
        button1.configure(command=lambda: self.color_change(stat, 0))
        button2.configure(command=lambda: self.color_change(stat, 1))

    def color_change(self, stat, i):
        if self.scale_buttons[stat][i].cget('bg') == self.scale_color:
            self.scale_buttons[stat][i].configure(bg='white')
        else:
            self.scale_buttons[stat][i].configure(bg=self.scale_color)
            self.scale_buttons[stat][(i + 1) % 2].configure(bg='white')

    # getting the opinion of each an expert (exp_num indicates which expert)
    def get_scale(self, exp_num):
        all = list(itertools.combinations(self.criteria, 2)) + self.subcriteria
        for stats_pair in all:
            chosen = self.scale_var[stats_pair].get()
            if not chosen:  # nothing in combobox
                self.incomplete_data = True
                if stats_pair not in self.subcriteria:
                    ind1 = self.criteria.index(stats_pair[1])
                    ind2 = self.criteria.index(stats_pair[0])
                    self.chosen_scale[exp_num, ind1, ind2] = None
                    self.chosen_scale[exp_num, ind2, ind1] = None
                else:
                    idx = self.subcriteria.index(stats_pair)  # which subcriteria
                    self.subscale[exp_num, idx, 0, 1] = None
                    self.subscale[exp_num, idx, 1, 0] = None

            elif self.scale_buttons[stats_pair][0].cget('bg') == self.scale_color:  # index 0 is chosen
                val = self.scale.index(chosen)
                val = val * 2 + 1
                if stats_pair not in self.subcriteria:
                    ind1 = self.criteria.index(stats_pair[0])
                    ind2 = self.criteria.index(stats_pair[1])
                    self.chosen_scale[exp_num, ind1, ind2] = val
                    self.chosen_scale[exp_num, ind2, ind1] = 1 / val
                else:
                    idx = self.subcriteria.index(stats_pair)  # which subcriteria
                    # the more important is the one at index 0 in tuple
                    self.subscale[exp_num, idx, 0, 1] = val
                    self.subscale[exp_num, idx, 1, 0] = 1 / val

            elif self.scale_buttons[stats_pair][1].cget('bg') == self.scale_color: # index 1 is chosen
                val = self.scale.index(chosen)
                val = val * 2 + 1
                if stats_pair not in self.subcriteria:
                    ind1 = self.criteria.index(stats_pair[1])
                    ind2 = self.criteria.index(stats_pair[0])
                    self.chosen_scale[exp_num, ind1, ind2] = val
                    self.chosen_scale[exp_num, ind2, ind1] = 1 / val
                else:
                    idx = self.subcriteria.index(stats_pair)
                    self.subscale[exp_num, idx, 0, 1] = 1 / val
                    self.subscale[exp_num, idx, 1, 0] = val

            else:   # no button is chosen
                self.incomplete_data = True
                if stats_pair not in self.subcriteria:
                    ind1 = self.criteria.index(stats_pair[1])
                    ind2 = self.criteria.index(stats_pair[0])
                    self.chosen_scale[exp_num, ind1, ind2] = None
                    self.chosen_scale[exp_num, ind2, ind1] = None
                else:
                    idx = self.subcriteria.index(stats_pair)
                    self.subscale[exp_num, idx, 0, 1] = None
                    self.subscale[exp_num, idx, 1, 0] = None

        # if all experts has spoken
        if (exp_num + 1) == self.experts:
            print("go to ranking")
            self.start_ranking()
        else: # cotinue opening scale windows
            self.open_scale_window(exp_num + 1)

    def open_options_window(self):
        self.delete_widgets()
        self.set_ok_button(3)

        cb = tk.Checkbutton(self.parent,
                            variable=self.varEVM,
                            text="EVM",
                            offvalue=0,
                            compound='left',
                            bg='white',
                            font=("Arial", 11)
                            )
        cb.place(x=100, y=50)

        cb = tk.Checkbutton(self.parent,
                            variable=self.varGMM,
                            text="GMM",
                            offvalue=0,
                            compound='left',
                            bg='white',
                            font=("Arial", 11)
                            )
        cb.place(x=100, y=100)

        label = tk.Label(self.parent,
                         text='Wpisz liczbę ekspertów',
                         bg='white',
                         font=("Arial", 11))
        label.place(x=400, y=50)

        self.entry = tk.Entry(self.parent,
                              )
        self.entry.place(x=400, y=100)

    def start_ranking(self):
        # which method is chosen
        if self.varEVM.get():
            self.method = 'EVM'
        elif self.varGMM.get():
            self.method = 'GMM'

        rank = Ranking(self.chosen_pokemons, self.chosen_scale, self.subscale, self.subcriteria, self.method,
                       self.incomplete_data, self.experts)
        self.ranking = rank.AHP()

        self.open_ranking_window()

    def open_ranking_window(self):
        self.delete_widgets()
        self.container = tk.Frame(self.parent, bg='white')
        self.canvas = tk.Canvas(self.container, width=self.size, height=self.size, bg='white', highlightthickness=0)
        self.add_scrollbar()

        numbers = []
        sorted_ranking = sorted(self.ranking, reverse=True) # sorting the ranking
        pokemon_images = []

        # displaying pokemons and the results
        for i in range(len(self.chosen_pokemons)):
            result = sorted_ranking[i]
            idx, = np.where(self.ranking == result)
            pokemon = self.chosen_pokemons[idx[0]]
            font_size = 13

            numbers.append(
                tk.Label(self.scrollable_frame, text="{}".format(i + 1), font=("Arial", font_size), bg='white', bd=0))
            numbers[i].grid(column=1, row=i + 1, padx=10)

            pokemon_images_label = tk.Label(self.scrollable_frame, image=self.checkboxes_images[pokemon.name], bd=0)
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

    def delete_widgets(self):
        for widget in self.parent.winfo_children():
            widget.destroy()


root = tk.Tk()
root.title("Pokemon Ranking")
gui = Gui(root)
root.mainloop()
