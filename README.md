

## Setup

Clone the repository:

```bash
git clone https://github.com/LapTQ/MCT.git
cd MCT
```

## Install dependencies

### Option 1: Docker

```bash
docker compose up
```

***Note***: You might need to [upgrade](https://docs.docker.com/engine/install/ubuntu/) your docker engine to the lastest version.

### Option 2: Conda

```bash
conda create --name MCT python==3.10.4
conda activate MCT
pip install -r requirements.txt
```

## Start application

If you're using docker, start the container:

```bash
docker start laptq_mct_backend -d
docker attach laptq_mct_backend
cd MCT
```

Else if you're using conda, activate the environment:

```bash
conda activate MCT
```

Then, run the application:

```bash
flask run
```

You can access the application via http://0.0.0.0:5555 from your browser.
