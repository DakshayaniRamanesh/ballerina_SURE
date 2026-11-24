import ballerina/http;
import ballerina/io;
import ballerina/time;

// First Ballerina client-wrapper milestone: conservative fixed reconciliation.
// V2-V5 are currently implemented in the Python reference engine, not this wrapper.
type Payment record {| string id; boolean keyed; |};
type Outcome record {| string id; string status; |};

function nowMs() returns decimal {
    time:Utc t = time:utcNow();
    return (<decimal>t[0] + t[1]) * 1000d;
}

class SureClient {
    private final string baseUrl;
    function init(string baseUrl) { self.baseUrl = baseUrl; }

    function execute(Payment payment, decimal deadlineMs) returns Outcome|error {
        decimal startTime = nowMs();
        decimal first = deadlineMs < 150d ? deadlineMs : 150d;
        if first <= 0d { return {id: payment.id, status: "unresolved"}; }
        http:Client original = check new (self.baseUrl, timeout = first / 1000d);
        Outcome|error response = original->post("/primary/payments", payment);
        if response is Outcome && response.status == "completed" && nowMs() - startTime <= deadlineMs {
            return response;
        }
        decimal remaining = deadlineMs - (nowMs() - startTime);
        if remaining <= 0d { return {id: payment.id, status: "unresolved"}; }
        http:Client statusClient = check new (self.baseUrl, timeout = remaining / 1000d);
        Outcome|error status = statusClient->get("/primary/payments/" + payment.id);
        if status is Outcome && status.status == "completed" && nowMs() - startTime <= deadlineMs {
            return status;
        }
        // Never repeat an unsafe operation; unknown is not proof of non-execution.
        return {id: payment.id, status: "unresolved"};
    }
}

public function main() returns error? {
    string url = "http://127.0.0.1:9090";
    string id = "demo-" + nowMs().toString();
    http:Client admin = check new (url);
    json configured = check admin->post("/admin/scenario", {id: id, mode: "lost", keyed: false});
    SureClient sure = new (url);
    Outcome result = check sure.execute({id: id, keyed: false}, 750d);
    io:println("Client outcome: ", result);
    json truth = check admin->get("/admin/truth/" + id);
    io:println("Independent ledger: ", truth);
}

