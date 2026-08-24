COMPOSE=docker compose
COUNT ?= 100

.PHONY: help demo up start down restart build rebuild ps logs logs-api logs-streamlit logs-grafana logs-prometheus logs-airflow logs-mlflow health traffic airflow-password airflow-reset-password clean reset-volumes

help:
	@echo "Commandes disponibles :"
	@echo "  make demo                  Lance toute la stack en arriere-plan avec rebuild"
	@echo "  make up                    Lance toute la stack avec logs visibles"
	@echo "  make start                 Lance toute la stack en arriere-plan"
	@echo "  make down                  Stoppe les conteneurs sans supprimer les volumes"
	@echo "  make restart               Redemarre la stack"
	@echo "  make build                 Rebuild les images"
	@echo "  make rebuild               Rebuild complet sans cache"
	@echo "  make ps                    Affiche les conteneurs"
	@echo "  make logs                  Affiche les logs de toute la stack"
	@echo "  make health                Teste les URLs principales"
	@echo "  make traffic COUNT=100     Genere du trafic de prediction"
	@echo "  make airflow-password      Affiche le mot de passe Airflow standalone"
	@echo "  make airflow-reset-password Force admin/admin pour Airflow"
	@echo "  make clean                 Stoppe et supprime les conteneurs"
	@echo "  make reset-volumes         Supprime conteneurs + volumes Docker"

demo:
	$(COMPOSE) down --remove-orphans
	$(COMPOSE) up -d --build
	$(COMPOSE) ps
	@echo ""
	@echo "Stack lancee."
	@echo ""
	@echo "URLs utiles :"
	@echo "  Streamlit   : http://127.0.0.1:8501"
	@echo "  FastAPI     : http://127.0.0.1:8001/docs"
	@echo "  MLflow      : http://127.0.0.1:5001"
	@echo "  Airflow     : http://127.0.0.1:8080"
	@echo "  Prometheus  : http://127.0.0.1:9090"
	@echo "  Grafana     : http://127.0.0.1:3000"
	@echo ""
	@echo "Grafana local : admin / admin"

up:
	$(COMPOSE) up --build

start:
	$(COMPOSE) up -d --build
	$(COMPOSE) ps

down:
	$(COMPOSE) down --remove-orphans

restart:
	$(COMPOSE) down --remove-orphans
	$(COMPOSE) up -d --build
	$(COMPOSE) ps

build:
	$(COMPOSE) build

rebuild:
	$(COMPOSE) down --remove-orphans
	$(COMPOSE) build --no-cache
	$(COMPOSE) up -d --force-recreate
	$(COMPOSE) ps

ps:
	$(COMPOSE) ps

logs:
	$(COMPOSE) logs -f --tail=100

logs-api:
	docker logs trustpilot-api -f --tail=100

logs-streamlit:
	docker logs trustpilot-streamlit -f --tail=100

logs-grafana:
	docker logs trustpilot-grafana -f --tail=100

logs-prometheus:
	docker logs trustpilot-prometheus -f --tail=100

logs-airflow:
	docker logs trustpilot-airflow -f --tail=100

logs-mlflow:
	docker logs trustpilot-mlflow -f --tail=100

health:
	@echo "FastAPI health:"
	curl.exe -f http://127.0.0.1:8001/health
	@echo ""
	@echo "Prometheus targets:"
	curl.exe -f http://127.0.0.1:9090/-/ready
	@echo ""
	@echo "Grafana:"
	curl.exe -f http://127.0.0.1:3000/api/health
	@echo ""
	@echo "Streamlit:"
	curl.exe -f http://127.0.0.1:8501

traffic:
	powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\generate_predictions.ps1 -Count $(COUNT)

airflow-password:
	docker exec trustpilot-airflow cat /opt/airflow/standalone_admin_password.txt

airflow-reset-password:
	docker exec trustpilot-airflow airflow users reset-password -u admin -p admin

clean:
	$(COMPOSE) down --remove-orphans

reset-volumes:
	$(COMPOSE) down -v --remove-orphans