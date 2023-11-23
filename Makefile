deploy_jupyter:
	docker compose -f docker-compose_jupyter.yml down
	docker build -f Dockerfile_jupyter -t metaglam_jupyter_environment .
	docker compose -f docker-compose_jupyter.yml up

jupyter_up:
	docker compose -f docker-compose_jupyter.yml up

jupyter_down:
	docker compose -f docker-compose_jupyter.yml down
