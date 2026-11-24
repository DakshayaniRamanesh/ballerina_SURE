import ballerina/http;
import ballerina/lang.runtime;
import ballerina/time;

// Synthetic, in-memory service. Both logical endpoints share a ledger.
// Admin endpoints are for experiments ONLY. Do not expose this service publicly.
type Scenario record {|
    string id;
    string mode;
    boolean keyed;
|};
type Payment record {|
    string id;
    boolean keyed;
|};
type Entry record {|
    string mode;
    boolean keyed;
    int attempts;
    int effects;
    int start_ms;
    int commit_ms;
|};
map<Entry> ledger = {};

function nowMs() returns int {
    time:Utc t = time:utcNow();
    return <int>((<decimal>t[0] + t[1]) * 1000d);
}

service / on new http:Listener(9090, {host: "127.0.0.1"}) {
    resource function get health() returns json {
        return {status: "ok"};
    }

    resource function get metrics() returns json {
        int effects = 0;
        int attempts = 0;
        int operations = 0;
        lock {
            operations = ledger.length();
            foreach Entry entry in ledger {
                effects += entry.effects;
                attempts += entry.attempts;
            }
        }
        return {logical_operations: operations, effects: effects, attempts: attempts};
    }

    resource function post admin/scenario(Scenario scenario) returns json {
        lock {
            ledger[scenario.id] = {mode: scenario.mode, keyed: scenario.keyed,
                attempts: 0, effects: 0, start_ms: 0, commit_ms: 0};
        }
        return {ok: true};
    }

    resource function get admin/truth/[string id]() returns json {
        lock {
            Entry? entry = ledger[id];
            if entry is Entry {
                return {mode: entry.mode, keyed: entry.keyed, attempts: entry.attempts,
                    effects: entry.effects, start_ms: entry.start_ms, commit_ms: entry.commit_ms};
            }
        }
        return {};
    }

    resource function post [string replica]/payments(Payment payment) returns json {
        string mode = "normal";
        int attempt = 0;
        boolean dropped = false;
        lock {
            Entry? entry = ledger[payment.id];
            if entry is () {
                return {status: "error", id: payment.id};
            }
            if entry.keyed != payment.keyed {
                return {status: "error", id: payment.id};
            }
            entry.attempts += 1;
            attempt = entry.attempts;
            mode = entry.mode;
            if mode == "primary_down" && replica == "primary" {
                return {status: "error", id: payment.id};
            }
            if entry.keyed && entry.effects > 0 {
                return {status: "completed", id: payment.id};
            }
            if entry.keyed && entry.start_ms > 0 {
                return {status: "pending", id: payment.id};
            }
            dropped = attempt == 1 && mode == "drop";
            if !dropped && entry.start_ms == 0 {
                entry.start_ms = nowMs();
            }
        }
        if dropped {
            runtime:sleep(0.32);
            return {status: "unknown", id: payment.id};
        }
        if attempt == 1 && mode == "slow" {
            runtime:sleep(0.32);
        } else {
            runtime:sleep(replica == "primary" ? 0.025 : 0.035);
        }
        lock {
            Entry? entry = ledger[payment.id];
            if entry is Entry {
                entry.effects += 1;
                if entry.commit_ms == 0 {
                    entry.commit_ms = nowMs();
                }
            }
        }
        if attempt == 1 && (mode == "lost" || mode == "status_slow" || mode == "status_down") {
            runtime:sleep(0.32);
        }
        return {status: "completed", id: payment.id};
    }

    resource function get [string replica]/payments/[string id]() returns json {
        string mode = "";
        lock {
            Entry? entry = ledger[id];
            if entry is Entry { mode = entry.mode; }
        }
        if mode == "status_down" { return {status: "error", id: id}; }
        runtime:sleep(mode == "status_slow" ? 0.25 : 0.045);
        string state = "unknown";
        lock {
            Entry? entry = ledger[id];
            if entry is Entry {
                if entry.effects > 0 { state = "completed"; }
                else if entry.start_ms > 0 { state = "pending"; }
            }
        }
        return {id: id, status: state};
    }
}
