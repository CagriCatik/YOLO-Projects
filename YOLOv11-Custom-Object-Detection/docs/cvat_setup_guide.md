# CVAT Setup & Dockerization Guide

This guide provides step-by-step instructions on how to set up CVAT (Computer Vision Annotation Tool) locally using Docker.

## 1. Prerequisites
- **Git**: [Download and install Git](https://git-scm.com/downloads).
- **Docker Desktop**: [Download and install Docker Desktop](https://www.docker.com/products/docker-desktop/).
  - **Important (Windows)**: Ensure WSL2 backend is enabled in Docker settings.
- **Google Chrome**: CVAT is optimized for Chrome.

## 2. Clone the Repository
Open a terminal (PowerShell or Git Bash) and execute:
```bash
git clone https://github.com/cvat-ai/cvat
cd cvat
```

## 3. Deployment with Docker Compose
Run the following command to download images and start containers:
```bash
docker compose up -d
```
*Note: This might take several minutes depending on your internet connection.*

## 4. Create a Superuser
You need an admin account to log in for the first time:
```bash
docker exec -it cvat_server python3 manage.py createsuperuser
```
Follow the prompts to enter a username, email, and password.

## 5. Accessing CVAT
1. Open Google Chrome.
2. Go to `http://localhost:8080`.
3. Log in with the superuser credentials you just created.

## 6. Exporting for YOLO
Once you have finished labeling your images in CVAT:
1. Go to your **Task** or **Project**.
2. Click on **Export Dataset**.
3. Select **YOLO 1.1** as the format.
4. Download the `.zip` file.
5. Extract it and place the content into your `datasets/custom_data/` folder as described in the `implementation_plan.md`.

## Troubleshooting
- **Memory**: CVAT requires at least 4GB (8GB recommended) of RAM allocated to Docker.
- **Ports**: If port 8080 is occupied, you'll need to change it in the `docker-compose.yml` or set an environment variable.
- **Updates**: To update CVAT, run `git pull` followed by `docker compose up -d --build`.
