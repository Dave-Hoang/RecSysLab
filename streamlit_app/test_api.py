from services.api_client import api_client

print(api_client.health())

response = api_client.recommend(
    query="psychological sci-fi movies"
)

print(response["recommendations"][0]["title"])