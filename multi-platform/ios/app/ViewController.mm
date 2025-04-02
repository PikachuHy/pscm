//
//  ViewController.m
//  demo
//
//  Created by PikachuHy on 2023/6/17.
//

#import "ViewController.h"
#import "Scheme.hpp"

@interface ViewController () {
  Scheme scm;
  __weak IBOutlet UILabel *text;
}

@end

@implementation ViewController

- (void)viewDidLoad {
  [super viewDidLoad];
  // Do any additional setup after loading the view.
  std::string version = scm.eval("(version)");
  NSString *s = [NSString stringWithUTF8String:version.c_str()];
  text.text = s;
}

@end
