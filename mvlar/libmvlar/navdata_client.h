#ifndef _NAVDATA_CLIENT_H_
#define _NAVDATA_CLIENT_H_

#include "navdata_client.h"
#include "config.h"
#include "navdata_common.h"
#include "thread.h"

#include <iostream>
#include <boost/log/trivial.hpp>
#include <boost/thread.hpp>
#include <boost/array.hpp>
#include <boost/asio.hpp>

using boost::asio::ip::udp;

class navdata_client{

    public:
        navdata_client();
        ~navdata_client();

        inline void Join() { service_thread->join(); }

    private:
        boost::asio::io_service                 io_service;
        udp::socket                             socket;
        udp::endpoint                           remote;
        udp::endpoint                           local;

        boost::thread*                          service_thread; 

        boost::array<char, NAVDATA_MAX_SIZE>    recv_buf;

        locked_queue<navdata_demo_t>            queue_navdata;

        void run_service();

        navdata_client(navdata_client&); // Empeche d'appeler le constructeur par default ?
};

#endif // _NAVDATA_CLIENT_H_
