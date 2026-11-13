import io,os,unittest,urllib.request,urllib.error
from types import SimpleNamespace
from unittest.mock import patch
import gemini_http2

class TransportTest(unittest.TestCase):
    def setUp(self):
        self.original=urllib.request.urlopen;gemini_http2.install()
    def tearDown(self):urllib.request.urlopen=self.original
    def invoke(self,status=200,code=0):
        req=urllib.request.Request('https://generativelanguage.googleapis.com/v1beta/test',data=b'{"unchanged":true}',headers={'x-goog-api-key':'FAKE_TEST_KEY'})
        def curl(command,**kw):
            self.assertNotIn('FAKE_TEST_KEY',str(command))
            self.assertEqual(kw['input'],req.data)
            self.assertNotIn('LD_LIBRARY_PATH',kw['env'])
            fd=int(command[command.index('--config')+1].rsplit('/',1)[1]);config=os.read(fd,4096)
            self.assertIn(b'FAKE_TEST_KEY',config)
            hfd=int(command[command.index('--dump-header')+1].rsplit('/',1)[1]);os.write(hfd,f'HTTP/2 {status}\r\nRetry-After: 7\r\n\r\n'.encode())
            return SimpleNamespace(returncode=code,stdout=b'{"ok":true}\n'+str(status).encode(),stderr=b'')
        with patch.object(gemini_http2.subprocess,'run',curl):return urllib.request.urlopen(req,timeout=90)
    def test_success_preserves_body_and_headers(self):
        with self.invoke() as response:
            self.assertEqual(response.status,200);self.assertEqual(response.read(),b'{"ok":true}')
    def test_retryable_http_error_preserves_status_header_and_body(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:self.invoke(status=429)
        self.assertEqual(caught.exception.code,429);self.assertEqual(caught.exception.headers['Retry-After'],'7')
        self.assertEqual(caught.exception.read(),b'{"ok":true}')
    def test_transport_failure_is_not_a_response(self):
        with self.assertRaises(urllib.error.URLError):self.invoke(code=28)

if __name__=='__main__':unittest.main()
