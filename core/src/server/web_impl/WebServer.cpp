// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <chrono>

#include <oatpp/network/server/Server.hpp>
#include <oatpp-swagger/Controller.hpp>

#include "server/web_impl/WebServer.h"
#include "server/web_impl/component/SwaggerComponent.hpp"
#include "server/web_impl/controller/WebController.hpp"

#include "server/Config.h"

namespace milvus {
namespace server {
namespace web {

void
WebServer::Start() {
    thread_ptr_ = std::make_shared<std::thread>(&WebServer::StartService, this);
}

void
WebServer::Stop() {
    StopService();

    if (thread_ptr_ != nullptr) {
        thread_ptr_->join();
        thread_ptr_ = nullptr;
    }
}

Status
WebServer::StartService() {
    oatpp::base::Environment::init();

    Config& config = Config::GetInstance();
    std::string port;
    Status status;

    status = config.GetServerConfigWebPort(port);

    {
        AppComponent components = AppComponent(std::stoi(port));
        SwaggerComponent swaggerComponent("192.168.1.57", std::stoi(port));

        auto doc_endpoints = oatpp::swagger::Controller::Endpoints::createShared();

        auto user_controller = WebController::createShared();

        /* create ApiControllers and add endpoints to router */
        OATPP_COMPONENT(std::shared_ptr<oatpp::web::server::HttpRouter>, router);
        user_controller->addEndpointsToRouter(router);

        doc_endpoints->pushBackAll(user_controller->getEndpoints());

        auto swaggerController = oatpp::swagger::Controller::createShared(doc_endpoints);
        swaggerController->addEndpointsToRouter(router);

        /* Get connection handler component */
        OATPP_COMPONENT(std::shared_ptr<oatpp::network::server::ConnectionHandler>, connection_handler);

        /* Get connection provider component */
        OATPP_COMPONENT(std::shared_ptr<oatpp::network::ServerConnectionProvider>, connection_provider);

        /* create server */
        auto server = oatpp::network::server::Server(connection_provider, connection_handler);

        std::thread stop_thread([&server, this] {
            while (!this->try_stop_.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }

            server.stop();
            OATPP_COMPONENT(std::shared_ptr<oatpp::network::ClientConnectionProvider>, client_provider);
            client_provider->getConnection();
        });

        // start synchronously
        server.run();

        connection_handler->stop();

        stop_thread.join();
    }

    oatpp::base::Environment::destroy();

    return Status::OK();
}

Status
WebServer::StopService() {
    try_stop_.store(true);

    return Status::OK();
}

}  // namespace web
}  // namespace server
}  // namespace milvus
