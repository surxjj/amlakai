import requests

x = 1
i = 0
while i ==0:
    url = "https://admin.apilproperties.com/api/properties"

    params = {
        "page": x
    }
    response = requests.get(url,params=params)
    if response.status_code == 200:
        data = response.json()
        print("✅ Response data:")
        print(data["data"])
        x += 1
        if data["data"] == []:
            print("There is no data")
            i = 1
        else:
            
            print(data["data"][0])
    else:

        print(f"❌ Failed with status code: {response.status_code}")
