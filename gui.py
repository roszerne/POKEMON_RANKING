import tkinter as tk
from tkinter import ttk
import requests, io, itertools, tkinter.messagebox
from PIL import ImageTk, Image
import data
import pandas as pd


class Gui():
    def __init__(self, parent):
        self.size = 600
        self.parent = parent
        self.parent.geometry("{}x{}".format(self.size, self.size))
        self.parent.config(bg='white')

        self.url_images = 'https://img.pokemondb.net/artwork/'
        self.dataframe = None

        self.container = tk.Frame(parent, bg='white')
        self.canvas = tk.Canvas(self.container, width=self.size, height=self.size, bg='white', highlightthickness=0)

        self.set_data(20)

        self.add_scrollbar()  # scrollbar jest tylko w 1. oknie bo potem sie psulo i po prostu robie tak zeby sie miescilo xd
        self.chosen_pokemons = []
        self.checkbuttons_var = {}
        self.checkbuttons_images = {}
        self.checkbuttons = {}

        # second window
        self.scale_buttons = {}
        self.scale_comboboxes = {}
        self.scale_var = {}
        self.chosen_scale = []
        self.scale_color = '#68C0DC'
        self.stats = ['Attack', 'Defense', 'HP', 'Speed']
        self.scale = ['Equal importance', 'Somewhat more important', 'Much more important', 'Very much important',
                      'Absolutely more important']

        self.set_checkboxes()
        self.set_ok_button(1)
        self.container.pack()
        self.canvas.pack(side="left", fill="both", expand=True)

    def set_data(self, how_many):
        data.data()
        df = pd.read_json('PokemonData.json')
        df = df.set_index(['#'])
        self.dataframe = df.head(how_many)

        '''
        no to trzeba ogarnac, bo dopiero teraz zobaczylam ze sie do pewnego momentu wyswietlaja bo robilam na malych liczbach
        no i nawet pikachu nie ma wiec cos jest nie tak
        '''

    def set_ok_button(self, i):
        ok_button = tk.PhotoImage(file="okbutton--1-.png")
        img_label = tk.Label(image=ok_button)
        img_label.image = ok_button
        # no to jest slabe te ify ale jak dawalam nazwe funkcji to sie psulo no wiec zostawiam
        if i == 1:
            button1 = tk.Button(self.parent, image=ok_button, command=lambda: self.get_pokemons(), borderwidth=0,
                                bg='white')
        else:
            button1 = tk.Button(self.parent, image=ok_button, command=lambda: self.get_scale(), borderwidth=0,
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
            # bo wtedy nie wiem jak strona wyglada
            if " " in name:
                continue

            image_url = 'https://img.pokemondb.net/artwork/' + name.lower() + '.jpg'
            response = requests.get(image_url)
            image = io.BytesIO(response.content)
            image = Image.open(image).resize((100, 100))
            pokemon_image = ImageTk.PhotoImage(image)

            var = tk.IntVar(0)
            self.checkbuttons_var[name] = var
            self.checkbuttons_images[name] = pokemon_image
            cb = tk.Checkbutton(self.scrollable_frame,
                                variable=var,
                                text="          " + name + "   ",
                                offvalue=0,
                                image=pokemon_image,
                                compound='left',
                                bg='white',
                                font=("Arial", 10)
                                )
            cb.grid(sticky="w")  # to keep it aligned
            self.checkbuttons[name] = cb

    #po nacisnieciu pierwszego okej
    def get_pokemons(self):
        for name in self.dataframe['Name']:
            if " " in name:
                continue
            if self.checkbuttons_var[name].get() and name not in self.chosen_pokemons:
                self.chosen_pokemons.append(name)
                print("Dodaj tego pokemona" + name)

        if len(self.chosen_pokemons) < 5 or len(self.chosen_pokemons) > 9:
            tkinter.messagebox.showinfo("Error", "Wrong number")
        else:
            self.open_scale_window()

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
        y_combobox = 10 + 83.5 * i  #no to jest wyliczone poprzez wielokrotne eksperymenty, to jest chyba najbardziej rowne jak moze byc xd
        scale_combobox.place(x=400, y=y_combobox)
        self.scale_var[stat] = selected
        self.scale_comboboxes[stat] = scale_combobox

    #przyciski z kryteriami
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
        self.scale_buttons[stat][(i + 1) % 2].configure(bg='white') #jak przyciskasz jeden to ten drugi sie zmienia

    # tworze tablice ktora przyda sie potem - (wazniejsze kryterium, mniej wazne, i to co wybrane ze skali)
    def get_scale(self):
        for stats_pair in itertools.combinations(self.stats, 2):

            chosen = self.scale_var[stats_pair].get()
            if not chosen:
                tkinter.messagebox.showinfo("Error", "Choose all")
                break

            elif self.scale_buttons[stats_pair][0].cget('bg') == self.scale_color:
                self.chosen_scale.append((stats_pair[0], stats_pair[1], chosen))

            elif self.scale_buttons[stats_pair][1].cget('bg') == self.scale_color:
                self.chosen_scale.append((stats_pair[1], stats_pair[0], chosen))

            else:
                tkinter.messagebox.showinfo("Error", "Choose all")
                break

        print(self.chosen_scale)

    def open_ranking_window(self):
        for widget in self.parent.winfo_children():
            widget.destroy()

        numbers = []
        ranking = []
        ranking_labels = []

        '''
        to jest na pewno do zrobienia, tak zeby jeszcze zdjecia bylo widac i wgl
        '''
        for i in range(len(self.chosen_pokemons)):
            coord_y = (i + 1) * 80
            font_size = 15
            numbers.append(tk.Label(text="{}".format(i + 1), font=("Arial", font_size), bg='white', relief='ridge'))
            numbers[i].place(x=20, y=coord_y)
            ranking.append("JAKIS POKEMON")

            ranking_labels.append(tk.Button(text=ranking[i], font=("Arial", font_size), bg='white'), )
            ranking_labels[i].place(x=70, y=coord_y)

    def delete_old_window(self):
        self.canvas.delete('all')
        for widget in self.parent.winfo_children():
            widget.destroy()


root = tk.Tk()

gui = Gui(root)
root.mainloop()
