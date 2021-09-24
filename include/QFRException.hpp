 #ifndef QFR_QFRException_HPP
 #define QFR_QFRException_HPP

 class QFRException : public std::runtime_error {
	std::string msg;
public:
	explicit QFRException(std::string  msg) : std::runtime_error("QFR Exception"), msg(std::move(msg)) { }

	const char *what() const noexcept override {
		return msg.c_str();
	}
};

#endif // QFR_QFRException_HPP
