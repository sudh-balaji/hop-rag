# Hopf Algebra Chat Application

## Quick Start

1. Create virtual environment:

```bash
python -m venv env

# On macOS/Linux
source env/bin/activate

# On Windows
.\env\Scripts\activate
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python main.py
```

4. Access FastAPI documentation:

- Swagger UI: http://localhost:8000/docs

## API Endpoints

Test the API via FastAPI's interactive docs at `/docs`:

POST `/chat`

```json
{
  "query": "your question here"
}
```

## Models Available

- TinyLlama/TinyLlama-1.1B-Chat-v1.0 (default)

Choose model by changing `model_name` in ChatModel initialization.
