import tkinter as tk
from tkinter import ttk
import requests, io
from PIL import ImageTk, Image
import data
import pandas as pd

url_images ='https://img.pokemondb.net/artwork/'

data.data()
df = pd.read_json('PokemonData.json')
df = df.set_index(['#'])
df = df.head(15)

root = tk.Tk()
root.geometry('420x400')
container = tk.Frame(root)
canvas = tk.Canvas(container, width = 400, height = 400)


scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)


names = []
checkbuttons_var = {}
checkbuttons_images = {}

#to tak zaczelam ale mi sie nie chce
scale = ['Equal importance', 'Somewhat more important', 'Much more important']
def new_window():
    canvas.delete('all')
    for widget in root.winfo_children():
        widget.destroy()

    HP_label = tk.Label(text = "HP").place(x = 20, y = 50)
    Attack_label = tk.Label(text = "Attack").place(x=20, y = 100)

    selected = tk.StringVar()
    scale_combobox = ttk.Combobox(root, textvariable = selected)
    scale_combobox['values'] = scale
    scale_combobox['state'] = 'readonly'
    scale_combobox.place(x = 50, y = 50)
    # scale_combobox.pack(fill = 'x', padx = 5, pady = 5)
    # scale_combobox.bind("selected", scale_selected)

#nie wiem czemu dodaje sie ostatni zawsze, w sensie zmienia sie jego var, ale ogarne
def cb_command(name):
    if not checkbuttons_var[name].get():
        checkbuttons_var[name].set(1)
    else:
        checkbuttons_var[name].set(0)

for name in df['Name']:
    #bo wtedy nie wiem jak strona wyglada
    if " " in name:
        continue
    img_url = 'https://img.pokemondb.net/artwork/' + name.lower() + '.jpg'
    response = requests.get(img_url)
    image = io.BytesIO(response.content)
    image = Image.open(image)
    image = image.resize((100, 100))
    pokemon_image = ImageTk.PhotoImage(image)

    var = tk.IntVar(0)
    checkbuttons_var[name] = var
    checkbuttons_images[name] = pokemon_image
    cb = tk.Checkbutton(scrollable_frame,
                        variable = var,
                        command = lambda: cb_command(name),
                        text="   " + name,
                        image = pokemon_image,
                        compound = 'left',
                        pady = 5)
    cb.config(font = ("Courier", 10))
    cb.pack()


chosen_pokemons = []
def get_pokemons():
    for name in df['Name']:
        if " " in name:
            continue
        if (checkbuttons_var[name].get()):
            chosen_pokemons.append(name)
            print("Dodaj tego pokemona")

    #no komunikaty takie lepsze
    if len(chosen_pokemons) < 2:
        print("za malo")
    elif len(chosen_pokemons) > 5:
        print("za duzo")
    else:
        new_window()


button1 = tk.Button(root, text = "Zatwierdź", command = lambda: get_pokemons())
button1.pack()


container.pack()
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

root.mainloop()








