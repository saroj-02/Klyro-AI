.PHONY: api ui test
api:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
ui:
	streamlit run frontend/streamlit_app.py
test:
	pytest -q
