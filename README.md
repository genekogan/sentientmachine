# Sentient Machine

To send requests to sentient machine:

    curl http://localhost:5656/run -k -d '{"question": "What is the nature of consciousness?", "password": "_SERVER_PASSWORD_HERE_"}'  -H 'Content-Type: application/json'

You will immediately receive back a token. You can check on its status like this:

    curl http://localhost:5656/fetch -k -d '{"token": "_TOKEN_ID_"}'  -H 'Content-Type: application/json'

This will give you a progress bar (stuck at 0) while its running. At some point, it will fetch you a link to download the finished data.
