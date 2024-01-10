# Web application for freshwater monitoring the effectiveness of interventions and mitigation actions in New Zealand

This repository contains the code for the web application and the data processing scripts for the [Monitoring Freshwater Improvement Actions](https://ourlandandwater.nz/project/monitoring-freshwater-improvement-actions/) project funded by [Our Land And Water](https://ourlandandwater.nz/).

## Running the web application
The web app has been built in [Docker](https://docs.docker.com/) and has been set up to be deployed on [Docker Swarm](https://docs.docker.com/engine/swarm/). 
After installing Docker and Docker Compose, you'll need to clone this repository:

```
git clone https://github.com/headwaters-hydrology/olw2-sc008
```

The Dockerfile, docker-compose.yml (for local testing), and the docker-swarm.yml (for deployment) sits in the [web_app path]().

Once you're in that directory, you can run docker-compose as normal. For example:

```
docker-compose up
```

Once it has downloaded the images and run the new containers, go to either localhost:8000 or 127.0.0.1:8000 on your browser to play with the web app.

To deploy the web application, you'll need a running Docker Swarm instance. See [DockerSwarm.rocks](https://dockerswarm.rocks) for a guide on how to set up a Docker Swarm instance.

Once you have a Docker Swarm instance, you'll probably need to modify the docker-swarm.yml based on your circumstances. Then you can deploy it like any normal Docker Swarm stack:

```
docker stack deploy -c docker-swarm.yml olw-app
```

## Running the processing scripts
All of the data processing is performed by python scripts. 
To run the python scripts, you'll need to install a python environment with the dependencies defined in the pyproject.toml file in the root directory. Then you'll need to download all of the [source data files]() to the data directory and extract the files to the data directory.

To ensure that rtree works, these need to be installed on linux:
https://github.com/Toblerity/rtree/issues/64#issuecomment-574377066
```
sudo apt install libspatialindex-dev python3-rtree
```

The main.py file in the processing directory contains the sequence of modules/functions to run to generate all of the results and assets for the web app. It is recommended to run short bits at a time as there are many scripts and they will take a long time to run.

## Additional documentation
Detailed documentation for the individual processes can be found here:
https://github.com/headwaters-hydrology/olw2-sc008/tree/main/web_app/docs




