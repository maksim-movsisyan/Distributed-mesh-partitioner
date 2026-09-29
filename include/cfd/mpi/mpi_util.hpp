// Small MPI helpers: block distribution and all-to-all for vectors of PODs.
#pragma once

#include <mpi.h>
#include <concepts>
#include <vector>
#include <cstddef>
#include <algorithm> 
#include <limits>

#include "cfd/mpi/log.hpp"

namespace cfd::mpi {
// Balanced block distribution of n elements over p parts (blcoks):
// displs[i]..displs[i+1] — rank i owns displs[i]..displs[i+1] (displs has size p+1).
template <std::integral integer>
std::vector<integer> block_displ(integer n, std::size_t p) {
    std::vector<integer> result(p + 1, 0);
    std::size_t num_el = static_cast<std::size_t>(n);

    for (std::size_t i = 0; i < p; ++i) {
        result[i + 1] = result[i] + static_cast<integer>((num_el / p) + ((num_el % p) > i ? 1 : 0));
    }

    return result;   
}

// Owner of element g under the block distribution.
template <std::integral integer>
int block_owner(integer g, const std::vector<integer>& displ) {
    if (g < displ.front() || g >= displ.back()) { return -1; }
    auto it = std::upper_bound(displ.begin(), displ.end(), g);

    return static_cast<int>(std::distance(displ.begin(), it)) - 1;
}

// Generic zero-overhead all-to-all packed transfer for trivially copyable types (deprecated).
template <typename T>
void alltoallv_packed_dep(
    MPI_Comm comm, int nprocs,
    const std::vector<int>& send_counts,
    const std::vector<T>& send_buf,
    std::vector<T>& recv_buf) {
        
    static_assert(std::is_trivially_copyable_v<T>, 
                  "T must be trivially copyable for raw MPI byte transfers!");

    const auto nprocs_sz = static_cast<std::size_t>(nprocs);
    if (send_counts.size() != nprocs_sz) {
        fatal(comm, "alltoallv_packed, send_counts size must match nprocs");
    }

    // 1. Collective exchange of element counts
    std::vector<int> recv_counts(nprocs_sz, 0);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, 
                 recv_counts.data(), 1, MPI_INT, comm);

    // 2. Single-pass computation of byte counts & byte displacements with overflow guards
    std::vector<int> send_bytes(nprocs_sz), recv_bytes(nprocs_sz);
    std::vector<int> sdispls_bytes(nprocs_sz), rdispls_bytes(nprocs_sz);

    std::size_t total_send_elems = 0;
    std::size_t total_recv_elems = 0;
    std::size_t curr_sdispl_bytes = 0;
    std::size_t curr_rdispl_bytes = 0;

    constexpr std::size_t kMpiMax = static_cast<std::size_t>(std::numeric_limits<int>::max());

    for (std::size_t i = 0; i < nprocs_sz; ++i) {
        const std::size_t sc = static_cast<std::size_t>(send_counts[i]);
        const std::size_t rc = static_cast<std::size_t>(recv_counts[i]);

        const std::size_t sb = sc * sizeof(T);
        const std::size_t rb = rc * sizeof(T);

        if (sb > kMpiMax || rb > kMpiMax || 
            curr_sdispl_bytes > kMpiMax || curr_rdispl_bytes > kMpiMax) {
            fatal(comm, "alltoallv_packed: Data size or displacement exceeds MPI INT_MAX limit.");
        }

        send_bytes[i] = static_cast<int>(sb);
        recv_bytes[i] = static_cast<int>(rb);
        sdispls_bytes[i] = static_cast<int>(curr_sdispl_bytes);
        rdispls_bytes[i] = static_cast<int>(curr_rdispl_bytes);

        total_send_elems += sc;
        total_recv_elems += rc;
        curr_sdispl_bytes += sb;
        curr_rdispl_bytes += rb;
    }

    if (total_send_elems != send_buf.size()) {
        mpi::fatal(comm, "alltoallv_packed: send_buf.size() does not match sum of send_counts.");
    }

    // 3. Allocate exact receive buffer
    recv_buf.resize(total_recv_elems);

    // 4. Perform collective exchange
    MPI_Alltoallv(
        send_buf.data(), send_bytes.data(), sdispls_bytes.data(), MPI_BYTE,
        recv_buf.data(), recv_bytes.data(), rdispls_bytes.data(), MPI_BYTE,
        comm
    );
}

// Generic zero-overhead all-to-all packed transfer for trivially copyable types.
template <typename T>
void alltoallv_packed(
    MPI_Comm comm, int nprocs,
    const std::vector<int>& send_counts,
    const std::vector<T>& send_buf,
    std::vector<T>& recv_buf) {
        
    static_assert(std::is_trivially_copyable_v<T>, 
                  "T must be trivially copyable for raw MPI transfers!");

    const auto nprocs_sz = static_cast<std::size_t>(nprocs);
    if (send_counts.size() != nprocs_sz) {
        fatal(comm, "alltoallv_packed, send_counts size must match nprocs");
    }

    // 1. Collective exchange of element counts
    std::vector<int> recv_counts(nprocs_sz, 0);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, 
                 recv_counts.data(), 1, MPI_INT, comm);

    // 2. Single-pass computation of element counts & element displacements with overflow guards
    std::vector<int> sdispls(nprocs_sz), rdispls(nprocs_sz);

    std::size_t total_send_elems = 0;
    std::size_t total_recv_elems = 0;
    std::size_t curr_sdispl = 0;
    std::size_t curr_rdispl = 0;

    constexpr std::size_t kMpiMax = static_cast<std::size_t>(std::numeric_limits<int>::max());

    for (std::size_t i = 0; i < nprocs_sz; ++i) {
        const std::size_t sc = static_cast<std::size_t>(send_counts[i]);
        const std::size_t rc = static_cast<std::size_t>(recv_counts[i]);

        if (sc > kMpiMax || rc > kMpiMax || 
            curr_sdispl > kMpiMax || curr_rdispl > kMpiMax) {
            fatal(comm, "alltoallv_packed: Element count or displacement exceeds MPI INT_MAX limit.");
        }

        sdispls[i] = static_cast<int>(curr_sdispl);
        rdispls[i] = static_cast<int>(curr_rdispl);

        total_send_elems += sc;
        total_recv_elems += rc;
        curr_sdispl += sc;
        curr_rdispl += rc;
    }

    if (total_send_elems != send_buf.size()) {
        mpi::fatal(comm, "alltoallv_packed: send_buf.size() does not match sum of send_counts.");
    }

    // 3. Allocate exact receive buffer
    recv_buf.resize(total_recv_elems);

    // 4. Create custom MPI datatype representing exactly sizeof(T) bytes
    MPI_Datatype mpi_element_type;
    MPI_Type_contiguous(static_cast<int>(sizeof(T)), MPI_BYTE, &mpi_element_type);
    MPI_Type_commit(&mpi_element_type);

    // 5. Perform collective exchange using our elements instead of raw MPI_BYTE
    MPI_Alltoallv(
        send_buf.data(), send_counts.data(), sdispls.data(), mpi_element_type,
        recv_buf.data(), recv_counts.data(), rdispls.data(), mpi_element_type,
        comm
    );

    // 6. Free the temporary datatype
    MPI_Type_free(&mpi_element_type);
}

// Global sum/min/max (short wrappers).
long long ll_sum(long long local);
long long ll_min(long long local);
long long ll_max(long long local);
double d_sum(double local);
double d_min(double local);
double d_max(double local);

}