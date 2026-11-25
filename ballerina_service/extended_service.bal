import ballerina/http;
import ballerina/lang.runtime;
import ballerina/time;

// Local synthetic fixture only. Schedules are per-endpoint call ordinal, not wall time.
type Fault record {|
    int before_ms;
    int after_ms;
    boolean down;
    boolean drop;
|};
type Scenario record {|
    string id;
    boolean keyed;
    int process_ms;
    Fault[] primary;
    Fault[] backup;
    Fault[] status;
|};
type Payment record {|
    string id;
    boolean keyed;
|};
type Entry record {|
    Scenario spec;
    int primary_calls;
    int backup_calls;
    int status_calls;
    int attempts;
    int active;
    int effects;
    int start_ms;
    int commit_ms;
|};
map<Entry> ledger = {};

function nowMs() returns int {
    time:Utc t = time:utcNow();
    return <int>((<decimal>t[0] + t[1]) * 1000d);
}
function delayMs(int duration) {
    runtime:sleep(<decimal>duration / 1000d);
}
function validSchedule(Fault[] schedule) returns boolean {
    if schedule.length() != 5 { return false; }
    foreach Fault f in schedule {
        if f.before_ms < 0 || f.before_ms > 350 || f.after_ms < 0 || f.after_ms > 500 {
            return false;
        }
    }
    return true;
}
function selectFault(Fault[] schedule, int index) returns Fault {
    // Index starts at zero; beyond the registered sequence, reuse its last element.
    return schedule[index < schedule.length() ? index : schedule.length() - 1];
}
function finish(string id) {
    lock {
        Entry? entry = ledger[id];
        if entry is Entry { entry.active -= 1; }
    }
}
service / on new http:Listener(9091, {host: "127.0.0.1"}) {
    resource function get health() returns json {
        return {status: "ok", fixture: "sure_extended_v1"};
    }
    resource function post admin/scenario(Scenario scenario) returns json {
        if scenario.process_ms < 0 || scenario.process_ms > 1000 ||
            !validSchedule(scenario.primary) || !validSchedule(scenario.backup) || !validSchedule(scenario.status) {
            return {ok: false};
        }
        lock {
            if ledger.hasKey(scenario.id) { return {ok: false}; }
            ledger[scenario.id] = {spec: scenario, primary_calls: 0, backup_calls: 0,
                status_calls: 0, attempts: 0, active: 0, effects: 0, start_ms: 0, commit_ms: 0};
        }
        return {ok: true};
    }
    resource function get admin/truth/[string id]() returns json {
        lock {
            Entry? entry = ledger[id];
            if entry is Entry {
                return {keyed: entry.spec.keyed, attempts: entry.attempts, effects: entry.effects,
                    start_ms: entry.start_ms, commit_ms: entry.commit_ms, active: entry.active,
                    primary_calls: entry.primary_calls, backup_calls: entry.backup_calls,
                    status_calls: entry.status_calls};
            }
        }
        return {};
    }
    resource function post [string replica]/payments(Payment payment) returns json {
        Fault fault = {before_ms: 0, after_ms: 0, down: false, drop: false};
        int processing = 0;
        if replica != "primary" && replica != "backup" { return {id: payment.id, status: "error"}; }
        lock {
            Entry? entry = ledger[payment.id];
            if entry is () { return {id: payment.id, status: "error"}; }
            if entry.spec.keyed != payment.keyed { return {id: payment.id, status: "error"}; }
            if replica == "primary" {
                fault = selectFault(entry.spec.primary, entry.primary_calls);
                entry.primary_calls += 1;
            } else {
                fault = selectFault(entry.spec.backup, entry.backup_calls);
                entry.backup_calls += 1;
            }
            entry.attempts += 1;
            entry.active += 1;
            processing = entry.spec.process_ms;
        }
        delayMs(fault.before_ms);
        string result = "unknown";
        boolean execute = false;
        if fault.down { result = "error"; }
        else if !fault.drop {
            // Atomic shared deduplication reservation across both logical replicas.
            lock {
                Entry? entry = ledger[payment.id];
                if entry is Entry {
                    if payment.keyed && entry.effects > 0 { result = "completed"; }
                    else if payment.keyed && entry.start_ms > 0 { result = "pending"; }
                    else {
                        if entry.start_ms == 0 { entry.start_ms = nowMs(); }
                        execute = true;
                    }
                }
            }
        }
        if execute {
            delayMs(processing);
            lock {
                Entry? entry = ledger[payment.id];
                if entry is Entry {
                    entry.effects += 1;
                    if entry.commit_ms == 0 { entry.commit_ms = nowMs(); }
                    result = "completed";
                }
            }
        }
        // Delayed responses can exceed client timeout; no actual packet manipulation.
        delayMs(fault.after_ms);
        finish(payment.id);
        return {id: payment.id, status: result};
    }
    resource function get [string replica]/payments/[string id]() returns json {
        Fault fault = {before_ms: 0, after_ms: 0, down: false, drop: false};
        lock {
            Entry? entry = ledger[id];
            if entry is () { return {id: id, status: "unknown"}; }
            fault = selectFault(entry.spec.status, entry.status_calls);
            entry.status_calls += 1;
            entry.active += 1;
        }
        delayMs(fault.before_ms);
        string result = "unknown";
        if fault.down { result = "error"; }
        else if !fault.drop {
            lock {
                Entry? entry = ledger[id];
                if entry is Entry {
                    if entry.effects > 0 { result = "completed"; }
                    else if entry.start_ms > 0 { result = "pending"; }
                }
            }
        }
        delayMs(fault.after_ms);
        finish(id);
        return {id: id, status: result};
    }
}
