import json
import urllib.request
import os

from pathlib import Path
from bs4 import BeautifulSoup

def get_pypi_package_names(default_address='https://pypi.org/simple/'):
    """Download PyPi package names

    Args:
        default_address (str, optional): url for pypi. Defaults to 'https://pypi.org/simple/'.

    Returns:
        list: pypi package names
    """
    print("Loading pypi names")
    pypi_package_names = []
    try:
        content = urllib.request.urlopen(default_address)
    except urllib.error.URLError as e:
        print("Parsing pypi went wrong due to: {}".format(e))
        return None
    soup = BeautifulSoup(content, "lxml")
    for a in soup.findAll('a', href=True):
        pypi_package_names.append(a.text)
    return pypi_package_names

def get_R_forge_package_names(default_address='https://r-forge.r-project.org/softwaremap/trove_list.php?cat=c&form_cat=307&page='):
    """Download R-forge package names

    Args:
        default_address (str, optional): url for R-forge. Defaults to 'https://r-forge.r-project.org/softwaremap/trove_list.php?cat=c&form_cat=307&page='.

    Returns:
        list: list of R-forge packages
    """
    print("Loading R packages")
    rforge_packages = []
    counter = 1
    prev_rforge_packages = []
    current_rforge_packages = []
    try:
        content = urllib.request.urlopen('{}{}'.format(default_address, counter))
    except urllib.error.URLError as e:
        print("Parsing R packages went wrong due to: {}".format(e))
        return None
    
    soup = BeautifulSoup(content, "lxml")
    main_div = soup.find("div", {"id": "maindiv"})
    for idx, tab in enumerate(soup.findAll("table",{"class":"fullwidth"})):
        if idx != 0:
            for ref in tab.findAll("a", href=True):
                if ref.text != '[Filter]' and ref.text.strip():
                    current_rforge_packages.append(ref.text.rstrip())

    while current_rforge_packages != prev_rforge_packages:
        if counter % 10 == 0:
            print("Currently at R forge page {}".format(counter))
        prev_rforge_packages = current_rforge_packages
        rforge_packages.extend(prev_rforge_packages)
        counter += 1
        current_rforge_packages = []

        try:
            content = urllib.request.urlopen('{}{}'.format(default_address, counter))
        except urllib.error.URLError as e:
            print("Parsing R packages went wrong on page {}: {}".format(counter, e))
            return rforge_packages

        soup = BeautifulSoup(content, "lxml")
        main_div = soup.find("div", {"id": "maindiv"})
        for idx, tab in enumerate(soup.findAll("table",{"class":"fullwidth"})):
            if idx == 0:
                pass
            else:
                for ref in tab.findAll("a", href=True):
                    if ref.text != '[Filter]' and ref.text.strip():
                        current_rforge_packages.append(ref.text.rstrip())
    return rforge_packages

def get_swMATH_software_names(default_address='https://swmath.org/?which_search=browse&sel=all&sortby=-rank&&page='):
    """Download list of SwMATH software names

    Args:
        default_address (str, optional): swMATH url. Defaults to 'https://swmath.org/?which_search=browse&sel=all&sortby=-rank&&page='.

    Returns:
        list: swMATH software names
    """
    print("Loading SwMATH names")
    swMATH_packages = []
    counter = 1
    prev_swMATH = []
    current_swMATH = []

    try:
        content = urllib.request.urlopen('{}{}'.format(default_address, counter))
    except urllib.error.URLError as e:
        print("Parsing swMATH names went wrong due to: {}".format(e))
        return None

    soup = BeautifulSoup(content, "lxml")
    for h1 in soup.findAll("h1"):
        current_swMATH.append(h1.text)
    current_swMATH.remove('swMATH')

    while current_swMATH != prev_swMATH:
        if counter % 10 == 0:
            print("Currently at swMATH page {}".format(counter))
        prev_swMATH = current_swMATH
        swMATH_packages.extend(prev_swMATH)
        counter += 1
        current_swMATH = []

        try:
            content = urllib.request.urlopen('{}{}'.format(default_address, counter), timeout=20)
            soup = BeautifulSoup(content, "lxml")
        except urllib.error.URLError as e:
            print("Parsing swMATH names went wrong on page {}: {}".format(counter, e))
            return swMATH_packages
        except:
            print("Probably a timeout in loading {}... trying again".format(counter))
            counter -= 1
        else:
            for h1 in soup.findAll("h1"):
                current_swMATH.append(h1.text)
            current_swMATH.remove('swMATH')
    return swMATH_packages

def get_CRAN_package_names(default_address='https://cran.r-project.org/web/packages/available_packages_by_name.html'):
    """Download CRAN package names

    Args:
        default_address (str, optional): url for CRAN. Defaults to 'https://cran.r-project.org/web/packages/available_packages_by_name.html'.

    Returns:
        list: CRAN package names
    """
    print("Loading CRAN packages")
    cran_packages = []
    try:
        content = urllib.request.urlopen(default_address)
    except urllib.error.URLError as e:
        print("Parsing pypi went wrong due to: {}".format(e))
        return None
    soup = BeautifulSoup(content, 'lxml')
    for a in soup.findAll('a', href=True):
        cran_packages.append(a.text)
    return cran_packages

def get_Bioconductor_package_names(default_address='https://www.bioconductor.org/packages/release/bioc/'):
    """Bioconductor package names

    Args:
        default_address (str, optional): url for Bioconductor. Defaults to 'https://www.bioconductor.org/packages/release/bioc/'.

    Returns:
        list: Bioconductor package names
    """
    print("Loading Bioconductor packages")
    bioconductor_packages = []
    try:
        content = urllib.request.urlopen(default_address)
    except urllib.error.URLError as e:
        print("Parsing pypi went wrong due to: {}".format(e))
        return None
    soup = BeautifulSoup(content, 'lxml')
    div = soup.find("div", {"id": "PageContent"})
    for row in div.find_all('table')[0].find_all('tr'):
        for a in row.findAll('a', href=True):
            bioconductor_packages.append(a.text)
    return bioconductor_packages

def get_Anaconda_package_names(repo='anaconda', default_address='https://anaconda.org/{}/repo?page='):
    """Get Anaconda package names

    Args:
        repo (str, optional): repo location for anaconda. Defaults to 'anaconda'.
        default_address (str, optional): url for anaconda. Defaults to 'https://anaconda.org/{}/repo?page='.

    Returns:
        list: Anaconda package names
    """
    conda_packages = []
    counter = 1
    prev_conda_packages = []
    current_conda_packages = []
    try:
        req = urllib.request.Request('{}{}'.format(default_address.format(repo), counter), headers={'User-Agent': 'Mozilla/5.0'})
        content = urllib.request.urlopen(req)
    except urllib.error.URLError as e:
        print("Parsing Anaconda went wrong due to: {}".format(e))
        return None
    soup = BeautifulSoup(content, 'lxml')
    for span in soup.findAll("span", {"class": "packageName"}):
        current_conda_packages.append(span.text)
    while current_conda_packages != prev_conda_packages:
        if (counter+1) % 1 == 0:
            print("Currently at Anaconda ({}) page {}".format(repo, counter+1))
        prev_conda_packages = current_conda_packages
        conda_packages.extend(prev_conda_packages)
        counter += 1
        current_conda_packages = []
        try:
            req = urllib.request.Request('{}{}'.format(default_address.format(repo), counter), headers={'User-Agent': 'Mozilla/5.0'})
            content = urllib.request.urlopen(req, timeout=20)
            soup = BeautifulSoup(content, 'lxml')
        except urllib.error.URLError as e:
            print("Parsing Anaconda names went wrong on page {}: {}".format(counter, e))
            return conda_packages
        except:
            print("Probably a timeout in loading {}... trying again".format(counter))
            counter -= 1
        else:
            for span in soup.findAll("span", {"class": "packageName"}):
                current_conda_packages.append(span.text)
    return conda_packages

def get_if_not_exists(date, function, name, location='/tmp', *args):
    """Load file if it exists, else call a function to create it.

    Args:
        date (str): string identifier of file
        function (fct): function to call is file does not exist
        name (str): base name of file
        location (str, optional): path to file. Defaults to '/tmp'.

    Returns:
        json-serializeable object: result from generating or loading a json file
    """
    loc = Path(location) / '{}_{}'.format(date, name)
    output = None
    if loc.is_file():
        print("Loading {} from {}".format(name, str(loc)))
        with loc.open(mode='r') as j_in:
            output = json.load(j_in)
    else:
        output = function(*args)
        with loc.open(mode='w') as j_out:
            json.dump(output, j_out, indent=4)
        print("Saved {} to {}".format(name, str(loc)))
    return output

def load_package_names(date, location='/tmp'):
    """Load package names from a number of repositories

    Args:
        date (str): date-based identifier
        location (str, optional): output/lookup path. Defaults to '/tmp'.

    Returns:
        dictionary: packages repositories with their respective software lists
    """
    data_dictionary = {
        'pypi': get_if_not_exists(date, get_pypi_package_names, 'pypi.json', location),
        'rforge': get_if_not_exists(date, get_R_forge_package_names, 'rforge.json', location),
        'swMATH': get_if_not_exists(date, get_swMATH_software_names, 'swMATH.json', location),
        'CRAN': get_if_not_exists(date, get_CRAN_package_names, 'CRAN.json', location),
        'Bioconductor': get_if_not_exists(date, get_Bioconductor_package_names, 'Bioconductor.json', location),
        'Anaconda': get_if_not_exists(date, get_Anaconda_package_names, 'Anaconda.json', location, 'anaconda'),
        'Conda-forge': get_if_not_exists(date, get_Anaconda_package_names, 'Conda-forge.json', location, 'conda-forge')
    }
    return data_dictionary
    