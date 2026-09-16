// This file is part of BOINC.
// http://boinc.berkeley.edu
//
// Custom balancing algorithm for BOINC

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sys/time.h>
#include <vector>

#include "boinc_db.h"

#include "buda.h"
#include "sched_check.h"
#include "sched_config.h"
#include "sched_custom_load_balancer.h"
#include "sched_keyword.h"
#include "sched_main.h"
#include "sched_msgs.h"
#include "sched_send.h"
#include "sched_shmem.h"
#include "sched_types.h"
#include "sched_score.h"
#include "sched_version.h"

namespace {

struct CUSTOM_JOB {
    JOB job;
    double predicted_seconds;
    double effective_flops;
};

double request_seconds_save[NPROC_TYPES];
double request_instances_save[NPROC_TYPES];

void clear_other_requests(int resource_type) {
    for (int i = 0; i < NPROC_TYPES; i++) {
        if (i == resource_type) {
            continue;
        }
        request_seconds_save[i] = g_wreq->req_secs[i];
        request_instances_save[i] = g_wreq->req_instances[i];
        g_wreq->req_secs[i] = 0;
        g_wreq->req_instances[i] = 0;
    }
}

void restore_other_requests(int resource_type) {
    for (int i = 0; i < NPROC_TYPES; i++) {
        if (i == resource_type) {
            continue;
        }
        g_wreq->req_secs[i] += request_seconds_save[i];
        g_wreq->req_instances[i] += request_instances_save[i];
    }
}

double clamp(double value, double low, double high) {
    return std::max(low, std::min(value, high));
}

// Returns CPUs number based on device type
// Using for modeling
double get_cpu_number() {
    const char* host = g_reply->host.domain_name;
    if (host && host[0]) {
        if (strstr(host, "cluster")) {
            return 4.0;
        }
        if (strstr(host, "desktop")) {
            return 2.0;
        }
        if (strstr(host, "low-power")) {
            return 1.0;
        }
        if (strstr(host, "phone")) {
            return 0.5;
        }
    }
    return std::max(0.5, static_cast<double>(g_reply->host.p_ncpus));
}

// esimated duration by BOINC divided by CPU number
double host_predicted_seconds(
    WORKUNIT& wu, BEST_APP_VERSION& bav
) {
    const double capacity = get_cpu_number();
    return estimate_duration(wu, bav) / capacity;
}

// custom random for determenistic behaviour between iterations
double random_custom(DB_ID_TYPE result_id, DB_ID_TYPE host_id) {
    unsigned long long value =
        static_cast<unsigned long long>(result_id)
        ^ (static_cast<unsigned long long>(host_id) << 32);
    value ^= value >> 33;
    value *= 0x1234567890abcdefULL;
    return static_cast<double>(value & 0xffffffULL)/0x1000000ULL;
}

// Score function, bases on baseline score
// Optimizes tasks distribution between devices
double custom_score(
    double baseline_score, const WORKUNIT& wu, double predicted_seconds,
    DB_ID_TYPE result_id
) {
    // increased baseline score for random, lpt, sjf policies
    const double protected_baseline_score = 100.0 * baseline_score;

    // logged job size for lpt & sjf policies
    const double runtime_order = std::log1p(std::max(0.0, predicted_seconds));

    // Resolving policy name
    if (!strcmp(config.custom_lb_policy, "random")) {
        // added random noise
        return protected_baseline_score + random_custom(result_id, g_reply->host.id);
    }
    if (!strcmp(config.custom_lb_policy, "lpt")) {
        // bigger for large tasks (log > 0)
        return protected_baseline_score + runtime_order;
    }
    if (!strcmp(config.custom_lb_policy, "sjf")) {
        // bigger for small tasks (log() < 0)
        return protected_baseline_score - runtime_order;
    }

    // target task executing timing (from config)
    const double target_runtime = std::max(1.0, config.custom_lb_target_runtime);

    // host strength (from config)
    const double host_speed = get_cpu_number();

    // BOINC already tracks hosts availability
    // for CHURN environment
    const double availability = clamp(g_wreq->cpu_available_frac, 0.15, 1.0);

    // predicted executing time based on availability
    const double effective_predicted = predicted_seconds / availability;

    // weight of task compared to target_runtime
    const double runtime_ratio = clamp(effective_predicted / target_runtime, 0.01, 100.0);

    // avoids sending large tasks to low-available hosts
    const double size_affinity = -std::fabs(std::log(runtime_ratio)) * availability;

    // bonus score for smaller tasks on low-available hosts
    const double deadline_pressure = target_runtime/std::max(1.0, static_cast<double>(wu.delay_bound));

    // penalty for large tasks
    const double runtime_penalty = std::log1p(std::max(0.0, effective_predicted) / target_runtime);

    // higher for smaller tasks on low-available hosts and large tasks on high-available hosts
    const double host_fit = (std::log1p(host_speed) - std::log1p(effective_predicted)) * availability * availability;

    // tuning penalty for large and medium tasks to smakller hosts
    // also helps to divide similar tasks between similar hosts
    double slow_host_long_penalty = 0.0;
    if (predicted_seconds > target_runtime * 0.5 && host_speed < 2.0) {
        slow_host_long_penalty = 1.5 * (predicted_seconds / target_runtime) * (2.0 - host_speed) / availability;
    } else if (predicted_seconds > target_runtime && host_speed < 4.0) {
        slow_host_long_penalty = 0.5 * (predicted_seconds / target_runtime - 1.0) * (4.0 - host_speed) / availability;
    }

    // scaling baseline score based on availability
    const double hybrid_baseline = (25.0 + 55.0 * availability) * baseline_score;

    return hybrid_baseline
        + config.custom_lb_size_weight * size_affinity
        + config.custom_lb_deadline_weight * deadline_pressure
        - config.custom_lb_runtime_weight * runtime_penalty
        + config.custom_lb_size_weight * 0.05 * host_fit
        - config.custom_lb_runtime_weight * slow_host_long_penalty;
}

// comparison method
bool custom_job_compare(const CUSTOM_JOB& left, const CUSTOM_JOB& right) {
    if (left.job.score == right.job.score) {
        return left.job.result_id < right.job.result_id;
    }
    return left.job.score > right.job.score;
}

long elapsed_microseconds(const timeval& start, const timeval& finish) {
    return (finish.tv_sec - start.tv_sec) * 1000000L + (finish.tv_usec - start.tv_usec);
}

void send_work_custom_type(int resource_type) {
    std::vector<CUSTOM_JOB> jobs;
    clear_other_requests(resource_type);

    const int random_offset = rand() % ssp->max_wu_results; // pure random offset

    for (int j = 0; j < ssp->max_wu_results; j++) {
        const int index = (j + random_offset) % ssp->max_wu_results;
        WU_RESULT& wu_result = ssp->wu_results[index];
        if (wu_result.state != WR_STATE_PRESENT && wu_result.state != g_pid) {
            continue;
        }

        WORKUNIT wu = wu_result.workunit;
        CUSTOM_JOB candidate;
        candidate.job.index = index;
        candidate.job.result_id = wu_result.resultid;
        candidate.job.app = ssp->lookup_app(wu.appid);
        if (!candidate.job.app || candidate.job.app->non_cpu_intensive) {
            continue;
        }

        const bool job_is_buda = is_buda(wu);
        candidate.job.bavp = get_app_version(wu, !job_is_buda, false);
        if (!candidate.job.bavp) {
            continue;
        }

        if (job_is_buda) {
            if (!choose_buda_variant(wu, resource_type, &candidate.job.buda_variant, candidate.job.host_usage)) {
                continue;
            }
        } else {
            candidate.job.host_usage = candidate.job.bavp->host_usage;
            candidate.job.buda_variant = NULL;
        }

        if (!candidate.job.get_score(index)) {
            continue;
        }

        if (job_is_buda) {
            HOST_USAGE& usage = candidate.job.host_usage;
            const double available = std::max(0.1, usage.uses_gpu() ? g_wreq->gpu_available_frac : g_wreq->cpu_available_frac);
            const double projected_flops = std::max(1.0, usage.projected_flops);
            candidate.effective_flops = projected_flops*available;
            candidate.predicted_seconds = std::max(0.0, wu.rsc_fpops_est) / candidate.effective_flops;
        } else {
            candidate.predicted_seconds = host_predicted_seconds(wu, *candidate.job.bavp);
            candidate.effective_flops = candidate.job.bavp->host_usage.projected_flops * get_cpu_number() * available_frac(*candidate.job.bavp);
        }
        candidate.job.score = custom_score(candidate.job.score, wu, candidate.predicted_seconds, candidate.job.result_id);
        jobs.push_back(candidate);
    }

    std::sort(jobs.begin(), jobs.end(), custom_job_compare);

    bool semaphore_locked = false;
    for (unsigned int i = 0; i < jobs.size(); i++) {
        if (!work_needed(false) || !g_wreq->need_proc_type(resource_type)) {
            break;
        }

        CUSTOM_JOB& candidate = jobs[i];
        JOB& job = candidate.job;
        if (config.max_jobs_in_progress.exceeded(job.app, job.bavp->host_usage.proc_type)) {
            continue;
        }
        if (daily_quota_exceeded(job.bavp)) {
            break;
        }

        if (!semaphore_locked) {
            lock_sema();
            semaphore_locked = true;
        }

        WU_RESULT& wu_result = ssp->wu_results[job.index];
        if ((wu_result.state != WR_STATE_PRESENT && wu_result.state != g_pid) || wu_result.resultid != job.result_id) {
            continue;
        }

        WORKUNIT wu = wu_result.workunit;
        if (wu_is_infeasible_fast(wu, wu_result.res_server_state, wu_result.res_priority, wu_result.res_report_deadline, *job.app, *job.bavp)) {
            continue;
        }
        wu_result.state = g_pid;
        unlock_sema();
        semaphore_locked = false;

        switch (slow_check(wu_result, job.app, job.bavp)) {
        case CHECK_NO_HOST:
            wu_result.state = WR_STATE_PRESENT;
            break;
        case CHECK_NO_ANY:
            wu_result.state = WR_STATE_EMPTY;
            if (config.keyword_sched) {
                keyword_sched_remove_job(job.index);
            }
            break;
        default:
            wu.hr_class = wu_result.workunit.hr_class;
            wu.app_version_id = wu_result.workunit.app_version_id;
            wu_result.state = WR_STATE_EMPTY;
            if (config.keyword_sched) {
                keyword_sched_remove_job(job.index);
            }

            SCHED_DB_RESULT result;
            result.id = wu_result.resultid;
            if (result_still_sendable(result, wu)) {
                add_result_to_reply(
                    result, wu, job.bavp, job.host_usage,
                    job.buda_variant, false
                );
                if (config.debug_custom_load_balancer) {
                    log_messages.printf(
                        MSG_NORMAL,
                        "[custom_lb] selected host_id=%lu result_id=%lu "
                        "policy=%s resource=%s score=%.6f predicted_seconds=%.3f "
                        "effective_flops=%.3f\n",
                        g_reply->host.id, job.result_id,
                        config.custom_lb_policy,
                        proc_type_name(resource_type), job.score,
                        candidate.predicted_seconds,
                        candidate.effective_flops
                    );
                }
            }
            break;
        }
    }

    if (semaphore_locked) {
        unlock_sema();
    }
    restore_other_requests(resource_type);
    g_wreq->best_app_versions.clear();
}

} // namespace

void send_work_custom() {
    timeval start, finish;
    gettimeofday(&start, NULL);
    const int jobs_before = g_wreq->njobs_sent;

    if (config.keyword_sched && g_request->user_keywords.empty()) {
        read_kw_prefs(g_request->user_id, g_request->user_keywords);
    }

    for (int i = NPROC_TYPES - 1; i >= 0; i--) {
        if (g_wreq->need_proc_type(i)) {
            send_work_custom_type(i);
        }
    }

    gettimeofday(&finish, NULL);
    if (config.debug_custom_load_balancer) {
        log_messages.printf(
            MSG_NORMAL,
            "[custom_lb] decision host_id=%lu policy=%s jobs_sent=%d "
            "elapsed_us=%ld\n",
            g_reply->host.id, config.custom_lb_policy,
            g_wreq->njobs_sent-jobs_before,
            elapsed_microseconds(start, finish)
        );
    }
}