//
//  Scheme.hpp
//  demo
//
//  Created by PikachuHy on 2023/6/17.
//

#ifndef Scheme_hpp
#define Scheme_hpp

#import <string>
class SchemeImpl;

class Scheme {
public:
  Scheme();
  ~Scheme();
  std::string eval(const char *code);

private:
  SchemeImpl *impl_;
};
#endif /* Scheme_hpp */
