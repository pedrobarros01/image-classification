ifneq ("$(wildcard .env)","")
	include .env
	export
endif

.PHONY: install
install: ## Install Python requirements.
	python -m pip install --upgrade pip setuptools wheel poetry
	poetry lock
	poetry install --no-root
	poetry run pre-commit install

.PHONY: plot
plot: ## Run the project.
	poetry run python -m src.app --modo "RES"

.PHONY: histr
histr: ## Run the project.
	poetry run python -m src.app --modo "EC" --tipo "hist_r"

.PHONY: histg
histg: ## Run the project.
	poetry run python -m src.app --modo "EC" --tipo "hist_g"

.PHONY: histb
histb: ## Run the project.
	poetry run python -m src.app --modo "EC" --tipo "hist_b"

.PHONY: msimples
msimples: ## Run the project.
	poetry run python -m src.app --modo "EC" --tipo "msimples"

.PHONY: mgeo
mgeo: ## Run the project.
	poetry run python -m src.app --modo "EC" --tipo "mgeo"

.PHONY: mhu
mhu: ## Run the project.
	poetry run python -m src.app --modo "EC" --tipo "mhu"

.PHONY: lbp
lbp: ## Run the project.
	poetry run python -m src.app --modo "EC" --tipo "lbp"

.PHONY: canny
canny: ## Run the project.
	poetry run python -m src.app --modo "EC" --tipo "canny"

.PHONY: knn-histr
knn-histr: ## Run the project.
	poetry run python -m src.app --modo "KNN" --tipo "hist_r"

.PHONY: knn-histg
knn-histg: ## Run the project.
	poetry run python -m src.app --modo "KNN" --tipo "hist_g"

.PHONY: knn-histb
knn-histb: ## Run the project.
	poetry run python -m src.app --modo "KNN" --tipo "hist_b"

.PHONY: knn-msimples
knn-msimples: ## Run the project.
	poetry run python -m src.app --modo "KNN" --tipo "msimples"

.PHONY: knn-mgeo
knn-mgeo: ## Run the project.
	poetry run python -m src.app --modo "KNN" --tipo "mgeo"

.PHONY: knn-mhu
knn-mhu: ## Run the project.
	poetry run python -m src.app --modo "KNN" --tipo "mhu"

.PHONY: knn-lbp
knn-lbp: ## Run the project.
	poetry run python -m src.app --modo "KNN" --tipo "lbp"

.PHONY: knn-canny
knn-canny: ## Run the project.
	poetry run python -m src.app --modo "KNN" --tipo "canny"

.DEFAULT_GOAL := help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sed 's/Makefile://g' | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
