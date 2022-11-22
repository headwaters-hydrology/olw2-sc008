# -*- coding: utf-8 -*-
# from .app import app, server
# from .layouts import layout1
# from . import callbacks

from app import app, server
from layouts import layout1
import callbacks

##########################################
### Parameters

app.layout = layout1

########################################
### run app


# if __name__ == '__main__':
#     app.run_server(host='0.0.0.0', port=80)


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8000)
