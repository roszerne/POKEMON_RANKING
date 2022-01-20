import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from ttkthemes import themed_tk as tk_theme
from ttkwidgets import Table

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

        self.set_data(20)

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
        self.clicked_buttons = {}

        self.ranking = []
        self.entry = None
        self.experts = 1  # number of experts, default is 1
        self.varMETHOD = tk.StringVar()
        self.incomplete_data = False
        self.incomplete_sub = False

        # create 1st window
        self.set_checkboxes()
        self.set_ok_button(1)
        self.container.pack()
        self.canvas.pack(side="left", fill="both", expand=True)

        self.s = ttk.Style()
        self.s.configure('.', font=('Helvetica', 13), background='white', focusthickness=3)

        self.ACTIVE_BUTTON = ttk.Style()
        self.ACTIVE_BUTTON.configure("ActiveButton.TButton", foreground='blue')

        self.BUTTON = ttk.Style()
        self.BUTTON.configure("Button.TButton", font=('Helvetica', 12), background='white')

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
        elif i == 2:  # after window with buttons and checkboxes, exp_num is the number of choosing expert
            button1 = tk.Button(self.parent, image=ok_button, command=lambda: self.get_scale(exp_num), borderwidth=0,
                                bg='white')
        else:  # after the option window, getting the number of experts and which method is to be used
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

        # columns =["Name"] + self.stats
        # table = Table(self.scrollable_frame, columns=columns, sortable=False, drag_cols=False,
        #               drag_rows=False, height = 20)
        # for col in columns:
        #     table.heading(col, text=col)
        #     table.column(col, width=100, stretch=False, anchor='center')

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

            cb = ttk.Checkbutton(self.scrollable_frame,
                                 variable=var,
                                 # text=self.get_checkbutton_text(i),
                                 image=pokemon_image
                                 )
            cb.grid(sticky="w")  # to keep it aligned
            self.checkboxes[name] = cb

        #     var = str(list(self.dataframe.to_records(index=False))[i]).strip('(')
        #     var.strip(")")
        #     table.insert('', 'end', values=var)
        #
        # table.grid()


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
        if not self.entry.get() or not self.entry.get().isdigit():
            messagebox.showerror("ERROR", "Wpisz odpowiednią liczbę ekspertów")
            return
        if not self.varMETHOD.get():
            messagebox.showerror("ERROR", "Wybierz metodę")
            return

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
        self.clicked_buttons[stat] = (False, False)
        button1 = ttk.Button(self.scrollable_frame, text=stat[0], style='Button.TButton')
        button2 = ttk.Button(self.scrollable_frame, text=stat[1], style='Button.TButton')
        button1.grid(column=3, row=i + 2, padx=20, pady=20)
        button2.grid(column=5, row=i + 2, padx=20, pady=20)
        self.scale_buttons[stat] = (button1, button2)
        button1.configure(command=lambda: self.click_button(0, stat))
        button2.configure(command=lambda: self.click_button(1, stat))

    def click_button(self, button, stat):
        var = self.clicked_buttons[stat][button]
        if not var:
            if button == 0:
                button_tuple = (True, False)
            else:
                button_tuple = (False, True)
            self.clicked_buttons[stat] = button_tuple
            self.scale_buttons[stat][button].configure(style="ActiveButton.TButton")
            self.scale_buttons[stat][(button + 1) % 2].configure(style='Button.TButton')
        else:
            self.clicked_buttons[stat] = (False, False)
            self.scale_buttons[stat][0].configure(style='Button.TButton')
            self.scale_buttons[stat][1].configure(style='Button.TButton')

    def get_scale(self, exp_num):
        all = list(itertools.combinations(self.criteria, 2)) + self.subcriteria
        for stats_pair in all:
            chosen = self.scale_var[stats_pair].get()
            if not chosen:  # nothing in combobox
                if stats_pair not in self.subcriteria:
                    self.incomplete_data = True
                    ind1 = self.criteria.index(stats_pair[1])
                    ind2 = self.criteria.index(stats_pair[0])
                    self.chosen_scale[exp_num, ind1, ind2] = None
                    self.chosen_scale[exp_num, ind2, ind1] = None
                else:
                    idx = self.subcriteria.index(stats_pair)  # which subcriteria
                    self.incomplete_sub = True
                    self.subscale[exp_num, idx, 0, 1] = None
                    self.subscale[exp_num, idx, 1, 0] = None

            elif self.clicked_buttons[stats_pair][0]:  # index 0 is chosen
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

            elif self.clicked_buttons[stats_pair][1]:  # index 1 is chosen
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

            else:  # no button is chosen
                if stats_pair not in self.subcriteria:
                    self.incomplete_data = True
                    ind1 = self.criteria.index(stats_pair[1])
                    ind2 = self.criteria.index(stats_pair[0])
                    self.chosen_scale[exp_num, ind1, ind2] = None
                    self.chosen_scale[exp_num, ind2, ind1] = None
                else:
                    idx = self.subcriteria.index(stats_pair)
                    self.incomplete_sub = True
                    self.subscale[exp_num, idx, 0, 1] = None
                    self.subscale[exp_num, idx, 1, 0] = None

        print(self.chosen_scale)
        # if all experts has spoken
        if (exp_num + 1) == self.experts:
            print("go to ranking")
            self.start_ranking()
        else:  # cotinue opening scale windows
            self.open_scale_window(exp_num + 1)

    def open_options_window(self):
        self.delete_widgets()
        self.set_ok_button(3)

        cb = ttk.Radiobutton(self.parent,
                             variable=self.varMETHOD,
                             value='EVM',
                             text="EVM",
                             compound='left'
                             )
        cb.place(x=250, y=100)

        cb = ttk.Radiobutton(self.parent,
                             variable=self.varMETHOD,
                             value='GMM',
                             text="GMM",
                             compound='left'
                             )

        cb.place(x=250, y=150)
        s = ttk.Style()
        s.configure('Entry.TLabel', font=('Helvetica', 14), background='white', focusthickness=3)

        label = ttk.Label(self.parent,
                          text='Wpisz liczbę ekspertów', style = 'Entry.TLabel')
        label.place(x=200, y=300)

        self.entry = ttk.Entry(self.parent, width = 20, font = ('Helvetica', 15))
        self.entry.place(x=190, y=350)

    def start_ranking(self):
        # which method is chosen
        self.method = self.varMETHOD.get()

        rank = Ranking(self.chosen_pokemons, self.chosen_scale, self.subscale, self.subcriteria, self.method,
                       self.incomplete_data, self.incomplete_sub, self.experts)
        self.ranking = rank.AHP()

        self.open_ranking_window()

    def open_ranking_window(self):
        self.delete_widgets()
        self.container = tk.Frame(self.parent, bg='white')
        self.canvas = tk.Canvas(self.container, width=self.size, height=self.size, bg='white', highlightthickness=0)
        self.add_scrollbar()

        numbers = []
        sorted_ranking = sorted(self.ranking, reverse=True)  # sorting the ranking
        pokemon_images = []

        # displaying pokemons and the results
        for i in range(len(self.chosen_pokemons)):
            result = sorted_ranking[i]
            idx, = np.where(self.ranking == result)
            pokemon = self.chosen_pokemons[idx[0]]
            font_size = 13

            numbers.append(
                ttk.Label(self.scrollable_frame, text="{}".format(i + 1)))
            numbers[i].grid(column=1, row=i + 1, padx=10)

            pokemon_images_label = ttk.Label(self.scrollable_frame, image=self.checkboxes_images[pokemon.name])
            pokemon_images.append(pokemon_images_label)
            pokemon_images[i].grid(column=2, row=i + 1, padx=10, pady=10)

            ranking_button = ttk.Button(self.scrollable_frame, text=pokemon.name + ':   ' + str(round(result, 3)))
            ranking_button.grid(column=3, row=i + 1, padx=10)

            stats_label = ttk.Label(self.scrollable_frame, text=pokemon.df[1:].to_string(),
                                    )
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


root = tk_theme.ThemedTk()
root.get_themes()
# root.set_theme("radiance")
root.set_theme("plastik")

root.title("Pokemon Ranking")
gui = Gui(root)
root.mainloop()
