//+------------------------------------------------------------------+
//|                                           HyprL_DataBridge.mq5   |
//|                                    HyprL Trading System          |
//|                           Data Bridge for Python Integration     |
//+------------------------------------------------------------------+
#property copyright "HyprL"
#property version   "2.00"
#property description "HTTP server providing historical data and order execution for Python"

#include <Trade\Trade.mqh>

//--- Input parameters
input int      ServerPort = 5555;              // HTTP Server Port
input string   AllowedOrigin = "*";            // CORS Origin
input int      MaxBarsPerRequest = 50000;      // Max bars per request
input bool     EnableTrading = true;           // Enable order execution
input string   ApiKey = "";                    // API Key (empty = no auth)

//--- Global variables
int serverSocket = INVALID_HANDLE;
CTrade trade;

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
    // Initialize trade object
    trade.SetExpertMagicNumber(123456);
    trade.SetDeviationInPoints(10);
    trade.SetTypeFilling(ORDER_FILLING_IOC);

    // Create server socket
    serverSocket = SocketCreate();
    if(serverSocket == INVALID_HANDLE)
    {
        Print("Error creating socket: ", GetLastError());
        return INIT_FAILED;
    }

    // Bind to port
    if(!SocketBind(serverSocket, "0.0.0.0", ServerPort))
    {
        Print("Error binding socket to port ", ServerPort, ": ", GetLastError());
        SocketClose(serverSocket);
        return INIT_FAILED;
    }

    // Listen for connections
    if(!SocketListen(serverSocket, 10))
    {
        Print("Error listening on socket: ", GetLastError());
        SocketClose(serverSocket);
        return INIT_FAILED;
    }

    Print("HyprL Data Bridge started on port ", ServerPort);
    Print("Endpoints: /health, /bars, /symbols, /account, /positions, /order");

    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    if(serverSocket != INVALID_HANDLE)
    {
        SocketClose(serverSocket);
        Print("HyprL Data Bridge stopped");
    }
}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{
    // Accept incoming connections
    int clientSocket = SocketAccept(serverSocket, 100);  // 100ms timeout

    if(clientSocket != INVALID_HANDLE)
    {
        HandleClient(clientSocket);
        SocketClose(clientSocket);
    }
}

//+------------------------------------------------------------------+
//| Handle HTTP client request                                         |
//+------------------------------------------------------------------+
void HandleClient(int clientSocket)
{
    // Read request
    char request[];
    int received = SocketRead(clientSocket, request, 4096, 1000);

    if(received <= 0) return;

    string requestStr = CharArrayToString(request, 0, received);

    // Parse HTTP request
    string method, path, query;
    ParseHttpRequest(requestStr, method, path, query);

    // Route request
    string response;

    if(path == "/health")
        response = HandleHealth();
    else if(path == "/bars")
        response = HandleBars(query);
    else if(path == "/symbols")
        response = HandleSymbols();
    else if(path == "/account")
        response = HandleAccount();
    else if(path == "/positions")
        response = HandlePositions();
    else if(path == "/order" && method == "POST")
        response = HandleOrder(requestStr);
    else
        response = JsonError("Unknown endpoint: " + path);

    // Send response
    SendHttpResponse(clientSocket, response);
}

//+------------------------------------------------------------------+
//| Parse HTTP request                                                 |
//+------------------------------------------------------------------+
void ParseHttpRequest(string request, string &method, string &path, string &query)
{
    // GET /bars?symbol=NVDA&timeframe=H1 HTTP/1.1
    string lines[];
    StringSplit(request, '\n', lines);

    if(ArraySize(lines) > 0)
    {
        string parts[];
        StringSplit(lines[0], ' ', parts);

        if(ArraySize(parts) >= 2)
        {
            method = parts[0];
            string fullPath = parts[1];

            int queryPos = StringFind(fullPath, "?");
            if(queryPos >= 0)
            {
                path = StringSubstr(fullPath, 0, queryPos);
                query = StringSubstr(fullPath, queryPos + 1);
            }
            else
            {
                path = fullPath;
                query = "";
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Get query parameter value                                          |
//+------------------------------------------------------------------+
string GetQueryParam(string query, string param)
{
    string pairs[];
    StringSplit(query, '&', pairs);

    for(int i = 0; i < ArraySize(pairs); i++)
    {
        string kv[];
        StringSplit(pairs[i], '=', kv);
        if(ArraySize(kv) >= 2 && kv[0] == param)
            return kv[1];
    }
    return "";
}

//+------------------------------------------------------------------+
//| Handle /health endpoint                                            |
//+------------------------------------------------------------------+
string HandleHealth()
{
    return "{\"status\":\"ok\",\"version\":\"2.0\",\"terminal\":\"" +
           TerminalInfoString(TERMINAL_NAME) + "\"}";
}

//+------------------------------------------------------------------+
//| Handle /bars endpoint                                              |
//+------------------------------------------------------------------+
string HandleBars(string query)
{
    string symbol = GetQueryParam(query, "symbol");
    string timeframe = GetQueryParam(query, "timeframe");
    string countStr = GetQueryParam(query, "count");

    if(symbol == "") return JsonError("Missing symbol parameter");
    if(timeframe == "") timeframe = "H1";

    int count = (int)StringToInteger(countStr);
    if(count <= 0) count = 1000;
    if(count > MaxBarsPerRequest) count = MaxBarsPerRequest;

    // Map timeframe string to ENUM_TIMEFRAMES
    ENUM_TIMEFRAMES tf = StringToTimeframe(timeframe);

    // Check if symbol exists
    if(!SymbolSelect(symbol, true))
    {
        return JsonError("Symbol not found: " + symbol);
    }

    // Get bars
    MqlRates rates[];
    int copied = CopyRates(symbol, tf, 0, count, rates);

    if(copied <= 0)
    {
        return JsonError("Failed to copy rates: " + IntegerToString(GetLastError()));
    }

    // Build JSON response
    string json = "{\"symbol\":\"" + symbol + "\",\"timeframe\":\"" + timeframe +
                  "\",\"count\":" + IntegerToString(copied) + ",\"bars\":[";

    for(int i = 0; i < copied; i++)
    {
        if(i > 0) json += ",";
        json += "{";
        json += "\"time\":" + IntegerToString((long)rates[i].time) + ",";
        json += "\"open\":" + DoubleToString(rates[i].open, 5) + ",";
        json += "\"high\":" + DoubleToString(rates[i].high, 5) + ",";
        json += "\"low\":" + DoubleToString(rates[i].low, 5) + ",";
        json += "\"close\":" + DoubleToString(rates[i].close, 5) + ",";
        json += "\"volume\":" + IntegerToString(rates[i].tick_volume);
        json += "}";
    }

    json += "]}";
    return json;
}

//+------------------------------------------------------------------+
//| Handle /symbols endpoint                                           |
//+------------------------------------------------------------------+
string HandleSymbols()
{
    string json = "{\"symbols\":[";

    int total = SymbolsTotal(false);
    bool first = true;

    for(int i = 0; i < total; i++)
    {
        string name = SymbolName(i, false);
        if(SymbolInfoInteger(name, SYMBOL_TRADE_MODE) != SYMBOL_TRADE_MODE_DISABLED)
        {
            if(!first) json += ",";
            json += "\"" + name + "\"";
            first = false;
        }
    }

    json += "]}";
    return json;
}

//+------------------------------------------------------------------+
//| Handle /account endpoint                                           |
//+------------------------------------------------------------------+
string HandleAccount()
{
    string json = "{";
    json += "\"balance\":" + DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2) + ",";
    json += "\"equity\":" + DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2) + ",";
    json += "\"margin\":" + DoubleToString(AccountInfoDouble(ACCOUNT_MARGIN), 2) + ",";
    json += "\"free_margin\":" + DoubleToString(AccountInfoDouble(ACCOUNT_MARGIN_FREE), 2) + ",";
    json += "\"leverage\":" + IntegerToString(AccountInfoInteger(ACCOUNT_LEVERAGE)) + ",";
    json += "\"currency\":\"" + AccountInfoString(ACCOUNT_CURRENCY) + "\",";
    json += "\"name\":\"" + AccountInfoString(ACCOUNT_NAME) + "\",";
    json += "\"server\":\"" + AccountInfoString(ACCOUNT_SERVER) + "\"";
    json += "}";
    return json;
}

//+------------------------------------------------------------------+
//| Handle /positions endpoint                                         |
//+------------------------------------------------------------------+
string HandlePositions()
{
    string json = "{\"positions\":[";

    int total = PositionsTotal();
    for(int i = 0; i < total; i++)
    {
        if(i > 0) json += ",";

        ulong ticket = PositionGetTicket(i);
        if(ticket > 0)
        {
            json += "{";
            json += "\"ticket\":" + IntegerToString(ticket) + ",";
            json += "\"symbol\":\"" + PositionGetString(POSITION_SYMBOL) + "\",";
            json += "\"type\":\"" + (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? "buy" : "sell") + "\",";
            json += "\"volume\":" + DoubleToString(PositionGetDouble(POSITION_VOLUME), 2) + ",";
            json += "\"open_price\":" + DoubleToString(PositionGetDouble(POSITION_PRICE_OPEN), 5) + ",";
            json += "\"current_price\":" + DoubleToString(PositionGetDouble(POSITION_PRICE_CURRENT), 5) + ",";
            json += "\"profit\":" + DoubleToString(PositionGetDouble(POSITION_PROFIT), 2) + ",";
            json += "\"sl\":" + DoubleToString(PositionGetDouble(POSITION_SL), 5) + ",";
            json += "\"tp\":" + DoubleToString(PositionGetDouble(POSITION_TP), 5);
            json += "}";
        }
    }

    json += "]}";
    return json;
}

//+------------------------------------------------------------------+
//| Handle /order endpoint (POST)                                      |
//+------------------------------------------------------------------+
string HandleOrder(string request)
{
    if(!EnableTrading)
        return JsonError("Trading is disabled");

    // Find JSON body (after empty line)
    int bodyStart = StringFind(request, "\r\n\r\n");
    if(bodyStart < 0) bodyStart = StringFind(request, "\n\n");
    if(bodyStart < 0) return JsonError("No request body");

    string body = StringSubstr(request, bodyStart + 4);

    // Parse JSON (simple parsing for known fields)
    string symbol = ExtractJsonString(body, "symbol");
    string action = ExtractJsonString(body, "action");
    double volume = ExtractJsonDouble(body, "volume");
    double sl = ExtractJsonDouble(body, "sl");
    double tp = ExtractJsonDouble(body, "tp");

    if(symbol == "") return JsonError("Missing symbol");
    if(action == "") return JsonError("Missing action");
    if(volume <= 0) volume = 0.01;

    // Execute order
    bool result = false;

    if(action == "buy")
    {
        double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
        result = trade.Buy(volume, symbol, ask, sl, tp, "HyprL");
    }
    else if(action == "sell")
    {
        double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
        result = trade.Sell(volume, symbol, bid, sl, tp, "HyprL");
    }
    else if(action == "close")
    {
        // Close all positions for symbol
        for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
            ulong ticket = PositionGetTicket(i);
            if(ticket > 0 && PositionGetString(POSITION_SYMBOL) == symbol)
            {
                trade.PositionClose(ticket);
            }
        }
        result = true;
    }
    else
    {
        return JsonError("Unknown action: " + action);
    }

    if(result)
    {
        return "{\"success\":true,\"ticket\":" + IntegerToString(trade.ResultOrder()) + "}";
    }
    else
    {
        return JsonError("Order failed: " + trade.ResultRetcodeDescription());
    }
}

//+------------------------------------------------------------------+
//| Convert timeframe string to enum                                   |
//+------------------------------------------------------------------+
ENUM_TIMEFRAMES StringToTimeframe(string tf)
{
    if(tf == "M1" || tf == "1m") return PERIOD_M1;
    if(tf == "M5" || tf == "5m") return PERIOD_M5;
    if(tf == "M15" || tf == "15m") return PERIOD_M15;
    if(tf == "M30" || tf == "30m") return PERIOD_M30;
    if(tf == "H1" || tf == "1h") return PERIOD_H1;
    if(tf == "H4" || tf == "4h") return PERIOD_H4;
    if(tf == "D1" || tf == "1d") return PERIOD_D1;
    if(tf == "W1" || tf == "1w") return PERIOD_W1;
    if(tf == "MN1" || tf == "1M") return PERIOD_MN1;
    return PERIOD_H1;
}

//+------------------------------------------------------------------+
//| Create JSON error response                                         |
//+------------------------------------------------------------------+
string JsonError(string message)
{
    return "{\"error\":\"" + message + "\"}";
}

//+------------------------------------------------------------------+
//| Extract string value from JSON                                     |
//+------------------------------------------------------------------+
string ExtractJsonString(string json, string key)
{
    string search = "\"" + key + "\":\"";
    int start = StringFind(json, search);
    if(start < 0) return "";

    start += StringLen(search);
    int end = StringFind(json, "\"", start);
    if(end < 0) return "";

    return StringSubstr(json, start, end - start);
}

//+------------------------------------------------------------------+
//| Extract double value from JSON                                     |
//+------------------------------------------------------------------+
double ExtractJsonDouble(string json, string key)
{
    string search = "\"" + key + "\":";
    int start = StringFind(json, search);
    if(start < 0) return 0;

    start += StringLen(search);
    string value = "";

    for(int i = start; i < StringLen(json); i++)
    {
        ushort c = StringGetCharacter(json, i);
        if((c >= '0' && c <= '9') || c == '.' || c == '-')
            value += ShortToString(c);
        else if(StringLen(value) > 0)
            break;
    }

    return StringToDouble(value);
}

//+------------------------------------------------------------------+
//| Send HTTP response                                                 |
//+------------------------------------------------------------------+
void SendHttpResponse(int socket, string body)
{
    string response = "HTTP/1.1 200 OK\r\n";
    response += "Content-Type: application/json\r\n";
    response += "Access-Control-Allow-Origin: " + AllowedOrigin + "\r\n";
    response += "Connection: close\r\n";
    response += "Content-Length: " + IntegerToString(StringLen(body)) + "\r\n";
    response += "\r\n";
    response += body;

    char responseArray[];
    StringToCharArray(response, responseArray, 0, WHOLE_ARRAY, CP_UTF8);

    SocketSend(socket, responseArray, ArraySize(responseArray) - 1);
}
//+------------------------------------------------------------------+
