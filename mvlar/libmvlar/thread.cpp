#include "thread.h"

// CONSTRUCTEUR

worker_pool::worker_pool(int pool_size = 0) : _work_io_service(_io_service), _thread_free(pool_size) {
	if (pool_size > 0) {
		// Si on donne pas de grosseur pool de thread on utilise le double du nombre de coeur du cpu
		pool_size = boost::thread::hardware_concurrency() * 2;

	}
}

worker_pool::~worker_pool() {
	_io_service.stop();
	try {
		_thread_group.join_all();
	}

}

// FONCTION PUBLIC

template < typename Job >
void worker_pool::run_job(Job j) {

}

// FONCTION PRIVER

void worker_pool::wrap_job(boost::function<void()> job) {
	
}
