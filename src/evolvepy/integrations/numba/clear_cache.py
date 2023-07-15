import os
import glob

import evolvepy

def clear_cache():

    evolvepy_path = os.path.dirname(evolvepy.__file__)

    for extension in ["*.nbc", "*.nbi"]:
        search_pattern = os.path.join(evolvepy_path, "**", extension)
        files = glob.glob(search_pattern, recursive=True)

        for f in files:
            os.remove(f)
    