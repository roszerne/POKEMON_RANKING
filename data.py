import requests
import lxml.html as lh
import pandas as pd


def make_header(tr_elements):
    col = []
    i = 0

    # For each row, store each first element (header) and an empty list
    for t in tr_elements[0]:
        i += 1
        name = t.text_content()
        col.append((name, []))
    return col


def get_data(tr_elements, col, how_many):
    # Since out first row is the header, data is stored on the second row onwards
    for j in range(1, how_many + 1):
        # T is our j'th row
        T = tr_elements[j]
        if j == 41 or j == 44 or j == 111 or j == 112:
            continue
        # If row is not of size 10, the //tr data is not from our table
        if len(T) != 10:
            break

        # i is the index of our column
        i = 0

        # Iterate through each element of the row
        for t in T.iterchildren():
            data = t.text_content()
            # Check if row is empty
            if i > 0:
                # Convert any numerical value to integers
                try:
                    data = int(data)
                except:
                    pass
            # Append the data to the empty list of the i'th column
            col[i][1].append(data)
            # Increment i for the next column
            i += 1


def data(how_many):
    url = 'http://pokemondb.net/pokedex/all'
    page = requests.get(url)
    page.encoding = 'utf-8'
    doc = lh.fromstring(page.content)
    tr_elements = doc.xpath('//tr')
    col = make_header(tr_elements)
    get_data(tr_elements, col, how_many)
    Dict = {title: column for (title, column) in col}
    df = pd.DataFrame(Dict)
    # df = df.drop(columns = ['Type', 'Total', 'Sp. Atk', 'Sp. Def']) # remove type and total column
    df = df.drop(columns=['Type', 'Total'])  # remove type and total column
    df.to_json('PokemonData.json')
